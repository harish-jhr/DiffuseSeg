import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
from tqdm import tqdm

from UNet import Unet
from noise_scheduler import NoiseScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureCaptureUnet(nn.Module):
    def __init__(self, original_unet: Unet, target_resolution=(64, 64)):
        super().__init__()
        self.unet = original_unet
        self.activations = {}
        self.target_resolution = target_resolution

        # def get_activation(name):
        #     def hook(model, input, output):
        #         attn_output_flat = output[0] 
        #         batch_size, seq_len, embed_dim = attn_output_flat.shape
                
        #         h = w = int(torch.sqrt(torch.tensor(seq_len).float()).item())
        #         if h * w != seq_len:
        #             raise ValueError(f"Feature map not square or seq_len mismatch: {seq_len}")

        #         attn_output_reshaped = attn_output_flat.transpose(1, 2).reshape(batch_size, embed_dim, h, w)
        #         self.activations[name] = attn_output_reshaped
        #     return hook

        # self.unet.ups[0].attentions[0].register_forward_hook(get_activation('ups0_attention_output'))
        # self.unet.ups[1].attentions[0].register_forward_hook(get_activation('ups1_attention_output'))
        # self.unet.ups[2].attentions[0].register_forward_hook(get_activation('ups2_attention_output'))

        def get_upblock_activation(name):
            def hook(module, input, output):
                # output is already (B, C, H, W)
                self.activations[name] = output.detach()
            return hook

        # register hooks on the UpBlock modules (capture final decoder outputs)
        self.unet.ups[0].register_forward_hook(get_upblock_activation('ups0_out'))
        self.unet.ups[1].register_forward_hook(get_upblock_activation('ups1_out'))
        self.unet.ups[2].register_forward_hook(get_upblock_activation('ups2_out'))

    def forward(self, x_noisy, t_val_tensor):
        self.activations = {}
        _ = self.unet(x_noisy, t_val_tensor) 
        
        all_upsampled_features = []
        #feature_keys_in_order = ['ups0_attention_output', 'ups1_attention_output', 'ups2_attention_output']
        feature_keys_in_order = ['ups0_out', 'ups1_out', 'ups2_out']

        for key in feature_keys_in_order:
            if key in self.activations:
                feature_map = self.activations[key]
                if feature_map.shape[2:] != self.target_resolution:
                    feature_map = F.interpolate(feature_map, size=self.target_resolution, mode='bilinear', align_corners=False)
                all_upsampled_features.append(feature_map)
        
        if all_upsampled_features:
            concatenated_features = torch.cat(all_upsampled_features, dim=1)
            return concatenated_features
        else:
            return None

#MAIN : 
if __name__ == "__main__":
    model_config = {
        'im_channels': 3,
        'im_size': 64,
        'down_channels': [64, 128, 256, 512],
        'mid_channels': [512, 512, 256],
        'down_sample': [True, True, True],
        'time_emb_dim': 256,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
        'num_heads': 4
    }

    diffusion_params = {
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02
    }

    scheduler = NoiseScheduler(
        num_timesteps=diffusion_params['num_timesteps'],
        beta_start=diffusion_params['beta_start'],
        beta_end=diffusion_params['beta_end']
    )
    
    model = Unet(model_config) 
    model.load_state_dict(torch.load("/dir/DiffuseSeg/weights/DDPM_Celeb_wts_2/DDPM_Celeb_151.pth"))
    model.eval() 
    model.to(device)

    target_img_H, target_img_W = 64, 64
    feature_extractor = FeatureCaptureUnet(model, target_resolution=(target_img_H, target_img_W))
    
    transform_image = T.Compose([
        T.ToTensor(),                           
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    selected_timesteps = [50, 150, 250] 
    
   
    data_dir = "/dir/DiffuseSeg/data/CelebAHQ256_Seg_Masks/64/"
    image_folder = os.path.join(data_dir, "images")
    
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
    
    num_training_images_to_extract = 100
    if num_training_images_to_extract is not None:
        image_filenames = image_filenames[:num_training_images_to_extract]

    all_pixel_features = []

    print(f"Starting feature extraction for {len(image_filenames)} images...")

   
    for img_filename in tqdm(image_filenames):
        img_path = os.path.join(image_folder, img_filename)

        x0_img = Image.open(img_path).convert("RGB")
        x0_tensor = transform_image(x0_img).unsqueeze(0).to(device) # (1, 3, H, W)

        # To fix noise for all timesteps for a given x0, as per paper
        # Use x0_tensor directly for torch.randn_like
        fixed_epsilon_for_x0 = torch.randn_like(x0_tensor).to(device)

        image_features_across_timesteps = []

        for t_val in selected_timesteps:
            #Fixed timesteps
            t_val_tensor = torch.tensor([t_val], dtype=torch.long, device=device)

            noisy_im = scheduler.add_noise(x0_tensor, fixed_epsilon_for_x0, t_val_tensor)

            with torch.no_grad():
                
                features_for_timestep = feature_extractor(noisy_im, t_val_tensor)
            
            if features_for_timestep is not None:
                image_features_across_timesteps.append(features_for_timestep)

        if image_features_across_timesteps:
            final_pixel_feature_map_for_x0 = torch.cat(image_features_across_timesteps, dim=1)
            # Flattening to (H*W, Total_Channels) and converting to float16 for efficiency
            flattened_features = final_pixel_feature_map_for_x0.permute(0, 2, 3, 1).reshape(-1, final_pixel_feature_map_for_x0.shape[1]).half()
            all_pixel_features.append(flattened_features)
        else:
            print(f"Warning: No features collected for image {img_filename} across any timestep.")

    print("\nFeature extraction complete. Concatenating all features...")

    if all_pixel_features:
        all_features_tensor = torch.cat(all_pixel_features, dim=0).cpu()

        print(f"Combined features tensor shape: {all_features_tensor.shape} (Dtype: {all_features_tensor.dtype})")

        output_dir = "./extracted_ddpm_features"
        os.makedirs(output_dir, exist_ok=True)
        features_save_path = os.path.join(output_dir, "ddpm_pixel_features_train.pt")

        torch.save(all_features_tensor, features_save_path)
        print(f"Features saved to: {features_save_path}")
    else:
        print("No features were collected. Check your data paths and extraction logic.")
