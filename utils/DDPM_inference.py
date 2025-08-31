import torch
import torchvision
import argparse
import yaml
import os
from tqdm import tqdm
from UNet import Unet
from noise_scheduler import NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, model_config, diffusion_config):
    # sampling by going from T timestep to 0 : saving X_0
    xt = torch.randn((infer_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
    for j in range(infer_config['num_samples']):
            img = ims[j]
            img = torchvision.transforms.ToPILImage()(img)
            img.save(os.path.join(infer_config['save_dir'], f"sample_{j+100}.png"))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='/dir/DiffuseSeg/DDPM/utils/config.yaml', type=str)
    args = parser.parse_args()
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    infer_config = config['infer_params']
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(infer_config['model_path'], map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = NoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler,  model_config, diffusion_config)