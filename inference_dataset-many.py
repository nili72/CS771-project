import torch
import numpy as np
import argparse

from sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange

from ldm.data.dataset2_augment import  Dataset2Eval, Dataset2Train

# NOTE: You have to be inside latent-diffusion folder in order to run the script
import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log



def ldm_cond_sample_dataset(config_path, ckpt_path, dataset, batch_size):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)
    logdir='/ocean/projects/mat240020p/nli1/diffusion/samples/2024-12-08T16-26-20_256_unet4_32_augment/images/'
    maskdir='/ocean/projects/mat240020p/nli1/diffusion/samples/2024-12-08T16-26-20_256_unet4_32_augment/masks/'

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_saved=0
    m_saved=0
    for batch_idx, x in enumerate(dataloader):
        real = x['image']
        real = rearrange(real, 'b h w c -> b c h w')
        seg = x['segmentation']
        masks=x['mask']
        for cond in masks:
            mask_image = Image.fromarray(cond.numpy())
            maskpath = os.path.join(maskdir, f"{m_saved:06}.png")
            mask_image.save(maskpath)
            m_saved += 1
       
        
        with torch.no_grad():
            seg = rearrange(seg, 'b h w c -> b c h w')

            condition = model.to_rgb(seg)
            seg = seg.to('cuda').float()
            seg = model.get_learned_conditioning(seg)
            samples, _ = model.sample_log(cond=seg, batch_size=batch_size, ddim=True,
                                      ddim_steps=200, eta=1.)
            samples = model.decode_first_stage(samples)
            for sample in samples:
                img = custom_to_pil(sample)
                imgpath = os.path.join(logdir, f"{n_saved:06}.png")
                img.save(imgpath)
                n_saved += 1
                
            print(n_saved)
            
            
            
            
def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved
'''
    save_image((real+1.0)/2.0, '/ocean/projects/mat240020p/nli1/diffusion/samples/512/real.png')
    save_image((samples+1.0)/2.0, '/ocean/projects/mat240020p/nli1/diffusion/samples/512/fake.png')
    save_image(condition, '/ocean/projects/mat240020p/nli1/diffusion/samples/512/cond.png')
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/ocean/projects/mat240020p/nli1/diffusion/results/2024-12-08T16-26-20_256_unet4_32_augment/configs/2024-12-08T16-26-20-project.yaml')#2024-11-15T01-00-29-project.yaml')
    parser.add_argument('--ckpt_path', type=str, default='/ocean/projects/mat240020p/nli1/diffusion/results/2024-12-08T16-26-20_256_unet4_32_augment/checkpoints/epoch=000012.ckpt')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    dataset = Dataset2Eval(size=256)# KvasirSegEval(size=256)
    ldm_cond_sample_dataset(args.config_path, args.ckpt_path, dataset, args.batch_size)
