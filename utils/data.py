import glob
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CelebHQDataset(Dataset):
    
    def __init__(self, split, im_path, im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.images = self.load_images(im_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),                          # [0,1]
            transforms.Normalize((0.5, 0.5, 0.5),           # shift to [-1,1]
                                 (0.5, 0.5, 0.5))
        ])

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"images path {im_path} does not exist"
        ims = sorted(glob.glob(os.path.join(im_path, f'*.{self.im_ext}')))
        print(f'Found {len(ims)} images for split {self.split}')
        return ims

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = Image.open(self.images[index]).convert("RGB")
        im_tensor = self.transform(im)
        return im_tensor
