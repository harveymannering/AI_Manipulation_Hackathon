"""Functions related to preference/reward models like ImageReward and HPSv2."""

import os
from abc import ABC, abstractmethod
from os.path import expanduser
from typing import List, Optional

import huggingface_hub
import ImageReward as RM
import open_clip
import torch
import torch.nn as nn
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import hps_version_map
from PIL import Image
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


class ScoreModel(ABC):
    """Abstract base class for all score models."""

    @abstractmethod
    def get_scores(
        self, image_paths: List[str], prompts: List[str] = []
    ) -> List[float]:
        """Calculate a score for the given image and (optionally) text prompt.

        Args:
            image_paths (List[str]): List of file path to the input images.
            prompts (List[str]): List of text prompts to evaluate with the images.

        Returns:
            List[float]: The computed scores.
        """
        pass


class CLIPScoreModel(ScoreModel):
    """CLIP Score used to calculate image-prompt alignment."""

    def __init__(self) -> None:
        """Initialize the CLIP model.

        Downloads the weights of CLIP and initialize the weights.
        """
        # Are we using a CPU or GPU?
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_scores(
        self, image_paths: List[str], prompts: List[str] = []
    ) -> List[float]:
        """Calculate the CLIP Score for an image-prompt pair.

        Args:
            image_paths (List[str]): File locations of the image.
            prompts (List[str]): Associated text prompts.

        Returns:
            List[float]: CLIP Scores.
        """
        # Load and preprocess images
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        ).to(self.device)

        # Forward pass to calculate CLIP Scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape (batch_size, 1)
        results: List[float] = logits_per_image.diagonal().tolist()
        return results


class AestheticScoreModel(ScoreModel):
    """Aesthetic scoring model using a pretrained linear model and OpenCLIP."""

    def __init__(self) -> None:
        """Initialize the aesthetic model.

        Downloads model weights if not already cached and loads
        the OpenCLIP encoder and preprocessing transforms.

        Raises:
            ValueError: If an unsupported CLIP model is specified.
        """
        clip_model = "vit_l_14"
        home = expanduser("~")
        cache_folder = os.path.join(home, ".cache/emb_reader")
        path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")

        # Download model weights if not already present
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
            from urllib.request import urlretrieve

            urlretrieve(url_model, path_to_model)

        # Initialize model architecture
        if clip_model == "vit_l_14":
            self.model = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            self.model = nn.Linear(512, 1)
        else:
            raise ValueError("Unsupported CLIP model")

        # Load weights and evaluation mode
        self.model.load_state_dict(torch.load(path_to_model))
        self.model.eval()

        # Load OpenCLIP model and preprocessing
        self.open_clip, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )

    def get_scores(
        self, image_paths: List[str], prompts: List[str] = []
    ) -> List[float]:
        """Calculate the Aesthetic score for an image.

        Args:
            image_paths (List[str]): File locations of the imagse.
            prompts (List[str]): Associated text prompts.

        Returns:
            List[float]: Aesthetic scores.
        """
        images = torch.stack(
            [self.preprocess(Image.open(path)) for path in image_paths]
        )
        with torch.no_grad():
            image_features = self.open_clip.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            scores = self.model(image_features)
        results: List[float] = scores.flatten().tolist()
        return results


class ImageRewardScoreModel(ScoreModel):
    """ImageReward model for scoring image-text alignment."""

    def __init__(self) -> None:
        """Load the ImageReward model.

        Uses pretrained checkpoint from ImageReward library.
        """
        self.model = RM.load("ImageReward-v1.0")

    def get_scores(
        self, image_paths: List[str], prompts: List[str] = []
    ) -> List[float]:
        """Calculate the ImageReward score for an image and a text prompt.

        Args:
            image_paths (List[str]): File locations of the images.
            prompts (List[str]): Associated text prompts.

        Returns:
            List[float]: ImageReward scores.
        """
        # Get scores from ImageReward
        _, rewards = self.model.inference_rank(prompts, image_paths)
        # Return the reward(s)
        if isinstance(rewards, float):
            return [rewards]
        else:
            # Get diagonal elements of the matrix
            n = int(len(rewards) ** 0.5)
            return [rewards[i * n + i] for i in range(n)]


class PickScoreModel(ScoreModel):
    """PickScore model for human preference prediction."""

    def __init__(self, device: Optional[str] = None) -> None:
        """Load the PickScore model and processor from HuggingFace.

        Args:
            device (str, optional): Manual set whether we use cpu or gpu. Defaults to None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_name = "yuvalkirstain/PickScore_v1"

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)
        self.device = device

    def get_scores(
        self, image_paths: List[str], prompts: List[str] = []
    ) -> List[float]:
        """Calculate the PickScore for an image and a text prompt.

        Args:
            image_paths (List[str]): File locations of the images.
            prompts (List[str]): Associated text prompts.

        Returns:
            List[float]: PickScore values.
        """
        with torch.no_grad():
            # Text embeddings
            text_inputs = self.processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # Image embeddings
            image_inputs = self.processor(
                images=[Image.open(img_path) for img_path in image_paths],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            # Similarity score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        results: List[float] = scores.flatten().tolist()
        return results


class HPSv2ScoreModel(ScoreModel):
    """HPSv2.1 model for human preference scoring."""

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize the HPSv2.1 model and preprocessing pipeline.

        Loads pretrained weights from HuggingFace.

        Args:
            device (str, optional): Manual set whether we use cpu or gpu. Defaults to None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create model and preprocessing transforms
        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=device,
            pretrained_image=False,
            output_dict=True,
            light_augmentation=True,
            aug_cfg={},
            with_score_predictor=False,
            with_region_predictor=False,
        )

        # Download checkpoint
        hps_version = "v2.1"
        checkpoint_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", hps_version_map[hps_version]
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        model.eval()

        self.model = model
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.preprocess = preprocess_val
        self.device = device

    def get_scores(
        self, image_paths: List[str], prompts: List[str] = []
    ) -> List[float]:
        """Calculate the HPSv2.1 score for an image and a text prompt.

        Args:
            image_paths (List[str]): File locations of the images.
            prompts (List[str]): Associated text prompts.

        Returns:
            List[float]: Human preference scores.
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            images = torch.stack(
                [self.preprocess(Image.open(path)) for path in image_paths]
            ).to(self.device)
            text = self.tokenizer(prompts).to(self.device)
            outputs = self.model(images, text)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            score = torch.diagonal(logits_per_image).cpu().numpy()
        results: List[float] = score.flatten().tolist()
        return results
