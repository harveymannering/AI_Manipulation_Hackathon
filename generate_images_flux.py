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
from diffusers import FluxKontextPipeline

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
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", 
        torch_dtype=torch.bfloat16,
        cache_dir = args.cache_dir
    )
    pipe.to(device)

    total_images = 100
    prompts = [
        "a portrait photo of a unattractive face",
        "a portrait photo of a face",
        "a portrait photo of a attractive face",
    ]
    
    # Iterate over paths and generate random expressions
    for img_idx in range(total_images):
        with torch.no_grad():
            latent = torch.randn(1, 4096, 64, device=pipe.transformer.device)
            latent = latent.repeat(len(prompts), 1, 1)
            images = pipe(
                latents=latent,
                prompt=prompts,
                num_inference_steps=20,
            ).images

            # Save images
            for p_idx, img in enumerate(images):
                if p_idx == 0:
                    output_word = "unattractive"
                elif p_idx == 1:
                    output_word = "normal"
                else:   
                    output_word = "attractive"
                img_path = os.path.join(args.output_dir, f"{img_idx}_{output_word}.jpg" )
                img.save(img_path)
