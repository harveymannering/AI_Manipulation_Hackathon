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
from utils import load_sdxl
import random
import numpy as np
import torch
from utils import ReNoise_Inversion
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

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
                img_path = os.path.join(args.output, f"{img_idx}_{p_idx}.jpg" )
                img.save(img_path)
