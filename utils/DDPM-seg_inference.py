import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

from UNet import Unet
from noise_scheduler import NoiseScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureCaptureUnet(nn.Module):
    def __init__(self, original_unet: Unet, target_resolution=(64, 64)):
        super().__init__()
        self.unet = original_unet
        self.activations = {}
        self.target_resolution = target_resolution

        def get_upblock_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        self.unet.ups[0].register_forward_hook(get_upblock_activation('ups0_out'))
        self.unet.ups[1].register_forward_hook(get_upblock_activation('ups1_out'))
        self.unet.ups[2].register_forward_hook(get_upblock_activation('ups2_out'))

    def forward(self, x_noisy, t_val_tensor):
        self.activations = {}
        _ = self.unet(x_noisy, t_val_tensor)
        all_upsampled_features = []
        for key in ['ups0_out', 'ups1_out', 'ups2_out']:
            if key in self.activations:
                feature_map = self.activations[key]
                if feature_map.shape[2:] != self.target_resolution:
                    feature_map = F.interpolate(feature_map, size=self.target_resolution, mode='bilinear', align_corners=False)
                all_upsampled_features.append(feature_map)
        if all_upsampled_features:
            return torch.cat(all_upsampled_features, dim=1)
        return None

class PixelMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        return self.fc_out(x)

def calculate_iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):
    iou_per_class = []
    predictions = predictions.cpu()
    targets = targets.cpu()
    for class_id in range(num_classes):
        tp = ((predictions == class_id) & (targets == class_id)).sum().item()
        fp = ((predictions == class_id) & (targets != class_id)).sum().item()
        fn = ((predictions != class_id) & (targets == class_id)).sum().item()
        denom = tp + fp + fn
        if denom == 0:
            iou = float('nan')
        else:
            iou = tp / denom
        iou_per_class.append(iou)
    valid_iou = [iou for iou in iou_per_class if not np.isnan(iou)]
    return (sum(valid_iou) / len(valid_iou)) if valid_iou else 0.0, iou_per_class

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

if __name__ == "__main__":
    MODEL_CONFIG = {
        'im_channels': 3, 'im_size': 64, 'down_channels': [64, 128, 256, 512],
        'mid_channels': [512, 512, 256], 'down_sample': [True, True, True],
        'time_emb_dim': 256, 'num_down_layers': 2, 'num_mid_layers': 2,
        'num_up_layers': 2, 'num_heads': 4
    }
    DIFFUSION_PARAMS = {'num_timesteps': 1000, 'beta_start': 0.0001, 'beta_end': 0.02}
    NUM_CLASSES = len(PART_COLORS)
    MLP_INPUT_DIM = 624
    NUM_MLPS_IN_ENSEMBLE = 10
    OUTPUT_DIR = "/dir/DiffuseSeg/extracted_ddpm_features/"
    SELECTED_TIMESTEPS = [50, 150, 250]

    scheduler = NoiseScheduler(**DIFFUSION_PARAMS)
    unet_model = Unet(MODEL_CONFIG)
    unet_model.load_state_dict(torch.load("/dir/DiffuseSeg/weights/DDPM_Celeb_wts_2/DDPM_Celeb_151.pth"))
    unet_model.eval().to(device)
    feature_extractor = FeatureCaptureUnet(unet_model, target_resolution=(MODEL_CONFIG['im_size'], MODEL_CONFIG['im_size']))

    mlps = []
    for i in range(1, NUM_MLPS_IN_ENSEMBLE + 1):
        mlp = PixelMLP(MLP_INPUT_DIM, NUM_CLASSES).to(device)
        mlp_path = os.path.join(OUTPUT_DIR, f"mlp_{i}_best.pt")
        if os.path.exists(mlp_path):
            mlp.load_state_dict(torch.load(mlp_path, map_location=device))
            mlp.eval()
            mlps.append(mlp)
    if not mlps:
        raise RuntimeError("No MLPs loaded.")

    transform_image = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_mask = T.Compose([
        T.Resize((MODEL_CONFIG['im_size'], MODEL_CONFIG['im_size']), T.InterpolationMode.NEAREST),
        T.ToTensor(), T.Lambda(lambda x: (x * 255).long()), T.Lambda(lambda x: x.squeeze(0))
    ])

    imgs_dir = "/dir/DiffuseSeg/weights/DDPM_Celeb_wts_2/samples_ep140/"
    masks_dir = "/dir/DiffuseSeg/weights/DDPM_Celeb_wts_2/samples_ep140/pred_masks__/"
    save_1_dir = "/dir/DiffuseSeg/weights/DDPM_Celeb_wts_2/samples_ep140/pred_masks/"
    # save_2_dir = "/dir/DiffuseSeg/data/CelebAHQ256_Seg_Masks/64/gt_masks_pred_151/"
    os.makedirs(save_1_dir, exist_ok=True)
    # os.makedirs(save_2_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(imgs_dir) if f.endswith(".png")]
    selected_imgs = random.sample(all_imgs, 150)

    for img_name in selected_imgs:
        test_image_path = os.path.join(imgs_dir, img_name)
        test_mask_path = os.path.join(masks_dir, img_name)

        original_x0_img = Image.open(test_image_path).convert("RGB")
        x0_tensor = transform_image(original_x0_img).unsqueeze(0).to(device)

        gt_mask_tensor = None
        if os.path.exists(test_mask_path):
            print(True)
            pil_mask = Image.open(test_mask_path).convert("RGB")
            mask_np_rgb = np.array(pil_mask).astype(np.uint8)
            gt_mask_tensor = torch.full((MODEL_CONFIG['im_size'], MODEL_CONFIG['im_size']), BACKGROUND_ID, dtype=torch.long)
            for color_rgb, class_id in color_to_id_map.items():
                matches = np.all(mask_np_rgb == np.array(color_rgb).astype(np.uint8), axis=-1)
                gt_mask_tensor[matches] = class_id
            gt_mask_tensor = gt_mask_tensor.to(device)

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
            test_pixel_features = final_pixel_feature_map_for_x0.permute(0, 2, 3, 1).reshape(-1, final_pixel_feature_map_for_x0.shape[1]).half().to(device)

        all_mlp_predictions = []
        with torch.no_grad():
            for mlp in mlps:
                outputs = mlp(test_pixel_features)
                _, predicted_class_ids = torch.max(outputs, 1)
                all_mlp_predictions.append(predicted_class_ids.cpu().numpy())
        stacked_predictions = np.stack(all_mlp_predictions, axis=0)
        ensemble_predicted_labels = [Counter(stacked_predictions[:, idx]).most_common(1)[0][0] for idx in range(stacked_predictions.shape[1])]
        predicted_mask_ids = torch.tensor(ensemble_predicted_labels, dtype=torch.long).reshape(MODEL_CONFIG['im_size'], MODEL_CONFIG['im_size']).to(device)

        predicted_rgb_mask = map_ids_to_colors(predicted_mask_ids, ID_TO_COLOR)
        save_path = os.path.join(save_1_dir, img_name)
        Image.fromarray(predicted_rgb_mask).save(save_path)
        print(f"Saved prediction: {save_path}")

        if gt_mask_tensor is not None:
            mIoU, _ = calculate_iou(predicted_mask_ids, gt_mask_tensor, NUM_CLASSES)
            print(f"\nMean IoU (mIoU) on test image: {mIoU:.4f}")
            # --- Visualization ---
            fig, axes = plt.subplots(1, 2 + (1 if gt_mask_tensor is not None else 0), figsize=(15, 6))
            
            # Original Image
            display_original_img = original_x0_img
            axes[0].imshow(display_original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Predicted Mask
            predicted_rgb_mask = map_ids_to_colors(predicted_mask_ids, ID_TO_COLOR)
            axes[1].imshow(predicted_rgb_mask)
            axes[1].set_title(f'Predicted Mask | mean IOU ({mIoU:.4f})')
            axes[1].axis('off')

            # Ground Truth Mask (if available)
            if gt_mask_tensor is not None:
                gt_rgb_mask = map_ids_to_colors(gt_mask_tensor, ID_TO_COLOR)
                axes[2].imshow(gt_rgb_mask)
                axes[2].set_title('Ground Truth Mask')
                axes[2].axis('off')
            save_path = os.path.join(save_2_dir, img_name)
            plt.tight_layout()
            plt.savefig(save_path)
        else:
            print("\nNo ground truth mask provided for evaluation.")
