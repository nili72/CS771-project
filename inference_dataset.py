import torch
import numpy as np
import argparse

from sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange

from ldm.data.kvasir import KvasirSegTrain, KvasirSegEval

# NOTE: You have to be inside latent-diffusion folder in order to run the script

def ldm_cond_sample_dataset(config_path, ckpt_path, dataset, batch_size):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x = next(iter(dataloader))

    real = x['image']
    real = rearrange(real, 'b h w c -> b c h w')
    seg = x['segmentation']

    with torch.no_grad():
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)
        print(condition)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                      ddim_steps=200, eta=1.)

        samples = model.decode_first_stage(samples)

    save_image((real+1.0)/2.0, '/ocean/projects/mat240020p/nli1/diffusion/samples/512/real.png')
    save_image((samples+1.0)/2.0, '/ocean/projects/mat240020p/nli1/diffusion/samples/512/fake.png')
    save_image(condition, '/ocean/projects/mat240020p/nli1/diffusion/samples/512/cond.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/ocean/projects/mat240020p/nli1/diffusion/results/2024-11-21T22-30-44_512_batch_1/configs/2024-11-21T22-30-44-project.yaml')#2024-11-15T01-00-29-project.yaml')
    parser.add_argument('--ckpt_path', type=str, default='/ocean/projects/mat240020p/nli1/diffusion/results/2024-11-21T22-30-44_512_batch_1/checkpoints/epoch=000001.ckpt')
    parser.add_argument('--batch_size', type=str, default=4)
    args = parser.parse_args()

    dataset = KvasirSegEval(size=512)
    ldm_cond_sample_dataset(args.config_path, args.ckpt_path, dataset, args.batch_size)
