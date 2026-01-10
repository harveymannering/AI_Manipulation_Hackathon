from __future__ import annotations
import numpy as np
import os
import torch
from torchvision import datasets, models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import pandas as pd
from PIL import Image
import cv2
import argparse
from diffusers import DiffusionPipeline
from sklearn.decomposition import PCA
import random
from typing import Callable
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
from tqdm import tqdm
import numpy as np
from diffusers import DDIMScheduler
import random
import numpy as np
import torch
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


def load_sdxl(args):
    
    # Create the DDIM noise schedule
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    # Initialize SDXL
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", # "SG161222/RealVisXL_V3.0", #"stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        use_safetensors=True,
        variant="fp16",
        cache_dir = args.cache_dir,
    )
    return pipe

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    
    # Set seed
    set_seed(0)
    
    # Load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Directory where we will save logging results.") 
    parser.add_argument("--cache_dir", required=True, help="Directory where the diffusion model is saved.") 
    args = parser.parse_args()
    
    # Initialize the generator network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = load_sdxl(args).to(device)
    total_images = 100
    prompts = [
        "a portrait photo of a unattractive face",
        "a portrait photo of a face",
        "a portrait photo of a attractive face",
    ]
    
    # Iterate over paths and generate random expressions
    for img_idx in range(total_images):
        with torch.no_grad():
            noise = torch.randn((1, 4, 64, 64)).to(device)
            noise = noise.repeat(total_images, 1, 1, 1)
            images = pipe(
                latents=noise.half(),
                prompts=prompts,
            ).images
            
            # Save images
            for p_idx, img in enumerate(images):
                if p_idx == 0:
                    output_word = "unattractive"
                elif p_idx == 1:
                    output_word = "normal"
                else:   
                    output_word = "attractive"
                img_path = os.path.join(args.output, f"{img_idx}_{output_path}.jpg" )
                img.save(img_path)
