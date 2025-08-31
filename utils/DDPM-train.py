import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from data import CelebHQDataset
from torch.utils.data import DataLoader
from UNet import Unet
from noise_scheduler import NoiseScheduler
import wandb
from sample import sample  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def train(args):
    # Reading config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    
    # wandb setup : 
    RUN_NAME = "train_03"         
    EXP_NAME = "DDPM-CelebHQ64" 

    wandb.init(
        project="DDPM-CelebHQ64",   
        name=RUN_NAME,            
        group=EXP_NAME,           
        force=True,              
    )


    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    celebhq = CelebHQDataset('train', im_path=dataset_config['im_path'])
    celebhq_loader = DataLoader(
        celebhq, batch_size=train_config['batch_size'], shuffle=True, num_workers=4
    )

    # Model
    model = Unet(model_config).to(device)
    model.train()

    if train_config['type'] == 'resume':
        model_path = train_config['model_path']
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training from scratch")

    # Optimizer, loss
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # Training loop
    global_step = 0
    for epoch_idx in range(num_epochs):
        losses = []
        for batch_idx, im in enumerate(tqdm(celebhq_loader)):
            optimizer.zero_grad()
            im = im.float().to(device)

            # Sample noise + timestep
            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # batchwise loss logging 
            wandb.log({
                "train/batch_loss": loss.item(),
                "step": global_step,
                "epoch": epoch_idx + 1,
                "batch": batch_idx + 1
            })
            global_step += 1

        # Epochwise loss logging
        mean_loss = np.mean(losses)
        print(f'Finished epoch:{epoch_idx + 1} | Loss : {mean_loss:.4f}')
        wandb.log({"train/epoch_loss": mean_loss, "epoch": epoch_idx + 1})

        # Saving checkpoints
        if epoch_idx % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(train_config['ckpt_path'], f'DDPM_Celeb_{epoch_idx + 1}.pth'))
        torch.save(model.state_dict(), os.path.join(train_config['ckpt_path'], 'DDPM_Celeb.pth'))

        # Sampling/Validation :
        if (epoch_idx + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                imgs = sample(model, scheduler, train_config, model_config, diffusion_config)
                wandb.log({
                    "validation/samples": [wandb.Image(imgs[i], caption=f"sample_{i}")
                                           for i in range(max(5, len(imgs)))]
                })
            model.train()

    print('Training Ends')
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='/dir/DiffuseSeg/DDPM/utils/config.yaml', type=str)
    args = parser.parse_args()
    train(args)
