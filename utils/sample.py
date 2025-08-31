import torch
import torchvision
import argparse
import yaml
import os
from tqdm import tqdm
from UNet import Unet
from noise_scheduler import NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, train_config, model_config, diffusion_config):
    # sampling by going from T timestep to 0 : saving X_0
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    imgs = []
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
    for j in range(train_config['num_samples']):
            img = ims[j]
            img = torchvision.transforms.ToPILImage()(img)
            imgs.append(img)
    return imgs
