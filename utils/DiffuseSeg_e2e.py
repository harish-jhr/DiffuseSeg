import torch
import os, sys
import yaml
import numpy as np
import random
from tqdm import tqdm
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from huggingface_hub import hf_hub_download


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'utils'))

from UNet import Unet
from noise_scheduler import NoiseScheduler
from Feature_extractor import FeatureCaptureUnet
from train_MLPs import PixelMLP, calculate_iou


#Color map
PART_COLORS = {
    "skin": (204, 204, 255), "l_brow": (255, 0, 0), "r_brow": (255, 0, 85),
    "l_eye": (255, 0, 170), "r_eye": (255, 0, 255), "eye_g": (170, 0, 255),
    "l_ear": (85, 0, 255), "r_ear": (0, 0, 255), "ear_r": (0, 85, 255),
    "nose": (0, 170, 255), "mouth": (0, 255, 255), "u_lip": (0, 255, 170),
    "l_lip": (0, 255, 85), "hair": (0, 255, 0), "hat": (170, 255, 0),
    "neck": (255, 255, 0), "cloth": (255, 170, 0), "necklace": (255, 85, 0),
    "earring": (170, 0, 85), "bg": (255, 255, 255)
}
ID_TO_COLOR, color_to_id_map = {}, {}
for idx, (name, rgb) in enumerate(PART_COLORS.items()):
    ID_TO_COLOR[idx] = rgb
    color_to_id_map[rgb] = idx
BACKGROUND_ID = color_to_id_map[PART_COLORS["bg"]]


def map_ids_to_colors(mask_ids: torch.Tensor, id_to_color_map: dict) -> np.ndarray:
    H, W = mask_ids.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id, color_rgb in id_to_color_map.items():
        rgb_mask[mask_ids.cpu().numpy() == class_id] = color_rgb
    return rgb_mask

#Function to sample using DDPM
def sample(model, scheduler, model_config, diffusion_config, device):
    xt = torch.randn((1,
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    
    with torch.no_grad():
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
    
    ims = torch.clamp(xt, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    img = ims[0] # The tensor for the single image
    pil_img = torchvision.transforms.ToPILImage()(img)
    return pil_img
    
#Run Segmentation using features from UNet
def run_inference_on_img(
    pil_img,
    feature_extractor,
    scheduler,
    mlps,
    MODEL_CONFIG,
    SELECTED_TIMESTEPS,
    ID_TO_COLOR,
    device
):
    transform_image = T.Compose([
        T.Resize((MODEL_CONFIG['im_size'], MODEL_CONFIG['im_size'])),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    x0_tensor = transform_image(pil_img).unsqueeze(0).to(device)
    fixed_epsilon_for_x0 = torch.randn_like(x0_tensor).to(device)
    image_features_across_timesteps = []
    with torch.no_grad():
        for t_val in SELECTED_TIMESTEPS:
            t_val_tensor = torch.tensor([t_val], dtype=torch.long, device=device)
            noisy_im = scheduler.add_noise(x0_tensor, fixed_epsilon_for_x0, t_val_tensor)
            features_for_timestep = feature_extractor(noisy_im, t_val_tensor)
            if features_for_timestep is not None:
                image_features_across_timesteps.append(features_for_timestep)
    
    final_pixel_feature_map_for_x0 = torch.cat(image_features_across_timesteps, dim=1)
    
    test_pixel_features = (
        final_pixel_feature_map_for_x0
        .permute(0, 2, 3, 1)  # (B,H,W,C)
        .reshape(-1, final_pixel_feature_map_for_x0.shape[1])  # (N,C)
        .half()
        .to(device)
    )
    
    all_mlp_predictions = []
    with torch.no_grad():
        for mlp in mlps:
            outputs = mlp(test_pixel_features)
            _, predicted_class_ids = torch.max(outputs, 1)
            all_mlp_predictions.append(predicted_class_ids.cpu().numpy())

    stacked_predictions = np.stack(all_mlp_predictions, axis=0)
    ensemble_predicted_labels = [
        Counter(stacked_predictions[:, idx]).most_common(1)[0][0]
        for idx in range(stacked_predictions.shape[1])
    ]
    
    predicted_mask_ids = torch.tensor(
        ensemble_predicted_labels, dtype=torch.long
    ).reshape(MODEL_CONFIG['im_size'], MODEL_CONFIG['im_size']).to(device)
    
    predicted_rgb_mask = map_ids_to_colors(predicted_mask_ids, ID_TO_COLOR)
    return predicted_mask_ids, predicted_rgb_mask

#Main bloc
if __name__ == "__main__":
    
    DEVICE_ID = 2
    device = f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    MODEL_CONFIG = config['model_params']
    DIFFUSION_PARAMS = config['diffusion_params']
    NUM_CLASSES = len(PART_COLORS)
    MLP_INPUT_DIM = 624 
    NUM_MLPS_IN_ENSEMBLE = 10
    SELECTED_TIMESTEPS = [50, 150, 250]

    #NoiseScheduler Initiation 
    scheduler = NoiseScheduler(**DIFFUSION_PARAMS)

    #Loading DDPM Weights from HF 
    print("Loading DDPM model")
    REPO_ID_DDPM = "Harish-JHR/DDPM_CelebAHQ64"
    FILENAME_DDPM = "DDPM_Celeb_151.pth"
    weights_path_ddpm = hf_hub_download(repo_id=REPO_ID_DDPM, filename=FILENAME_DDPM)
    unet_model = Unet(MODEL_CONFIG)
    unet_model.load_state_dict(torch.load(weights_path_ddpm, map_location=device))
    unet_model.eval().to(device)

    #Feature Extractor Initiation 
    feature_extractor = FeatureCaptureUnet(unet_model, target_resolution=(MODEL_CONFIG['im_size'], MODEL_CONFIG['im_size']))
    
    #Loading the weights of MLPs from HF 
    print("Loading MLP models for segmentation")
    REPO_ID_MLP = "Harish-JHR/DiffuseSegWeights"
    mlps = []
    for i in range(1, NUM_MLPS_IN_ENSEMBLE + 1):
        filename = f"mlp_{i}_best.pt"
        try:
            weight_path = hf_hub_download(repo_id=REPO_ID_MLP, filename=filename)
            mlp = PixelMLP(MLP_INPUT_DIM, NUM_CLASSES).to(device)
            mlp.load_state_dict(torch.load(weight_path, map_location=device))
            mlp.eval()
            mlps.append(mlp)
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Could not load {filename}: {e}")
    
    if not mlps:
        raise RuntimeError("Error : No MLPs loaded from Hugging Face repo")

    #Inferenvce
    print("Generating image with DDPM")
    ddpm_sample_img = sample(unet_model, scheduler, MODEL_CONFIG, DIFFUSION_PARAMS, device)
    
    print("Running segmentation on generated img")
    mask_ids, mask_rgb = run_inference_on_img(
        ddpm_sample_img, feature_extractor, scheduler, mlps,
        MODEL_CONFIG, SELECTED_TIMESTEPS, ID_TO_COLOR, device
    )
    print("Saving output image")
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(ddpm_sample_img)
    axes[0].set_title("Sampled using DDPM")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")
    
    output_path = "./DiffuseSeg_e2e.png"
    plt.savefig(output_path)
    print(f"Output at {output_path}")
