import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Custom Colormap : 
PART_COLORS = {
    "skin": (204, 204, 255),
    "l_brow": (255, 0, 0),
    "r_brow": (255, 0, 85),
    "l_eye": (255, 0, 170),
    "r_eye": (255, 0, 255),
    "eye_g": (170, 0, 255),
    "l_ear": (85, 0, 255),
    "r_ear": (0, 0, 255),
    "ear_r": (0, 85, 255),
    "nose": (0, 170, 255),
    "mouth": (0, 255, 255),
    "u_lip": (0, 255, 170),
    "l_lip": (0, 255, 85),
    "hair": (0, 255, 0),
    "hat": (170, 255, 0),
    "neck": (255, 255, 0),
    "cloth": (255, 170, 0),
    "necklace": (255, 85, 0),
    "earring": (170, 0, 85),
    "bg": (255, 255, 255)  # white background
}

# Integer Labels
color_to_id_map = {}
id_counter = 0
for part_name, color_rgb in PART_COLORS.items():
    color_to_id_map[color_rgb] = id_counter
    id_counter += 1
background_id = color_to_id_map[PART_COLORS["bg"]] 

target_img_H, target_img_W = 64, 64 

transform_mask_to_tensor = T.Compose([
    T.Resize((target_img_H, target_img_W), T.InterpolationMode.NEAREST), 
    T.ToTensor(), # Converts to (C, H, W) float in [0,1]
])


data_dir = "/dir/DiffuseSeg/data/CelebAHQ256_Seg_Masks/64/"
mask_folder = os.path.join(data_dir, "masks")

mask_filenames = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png")])

num_images_to_process = 100 
if num_images_to_process is not None:
    mask_filenames = mask_filenames[:num_images_to_process]

all_pixel_labels = []

print(f"Starting label extraction for {len(mask_filenames)} masks...")


for mask_filename in tqdm(mask_filenames):
    mask_path = os.path.join(mask_folder, mask_filename)

    pil_mask = Image.open(mask_path).convert("RGB")
    
    mask_tensor_rgb = (np.array(pil_mask)).astype(np.uint8) # (H,W,3) numpy array
    
    current_image_labels = torch.full((target_img_H, target_img_W), background_id, dtype=torch.long)


    for color_rgb, class_id in color_to_id_map.items():
        matches = np.all(mask_tensor_rgb == np.array(color_rgb).astype(np.uint8), axis=-1)
        current_image_labels[matches] = class_id
    all_pixel_labels.append(current_image_labels.flatten())

print("\nLabel extraction complete. Concatenating all labels : =")
if all_pixel_labels:
    all_labels_tensor = torch.cat(all_pixel_labels, dim=0).cpu()

    print(f"Combined labels tensor shape: {all_labels_tensor.shape} (Dtype: {all_labels_tensor.dtype})")

    output_dir = "./extracted_ddpm_features/" 
    os.makedirs(output_dir, exist_ok=True)
    labels_save_path = os.path.join(output_dir, "ddpm_pixel_labels_train.pt")

    torch.save(all_labels_tensor, labels_save_path)
    print(f"Labels saved to: {labels_save_path}")
else:
    print("No labels were collected.")
