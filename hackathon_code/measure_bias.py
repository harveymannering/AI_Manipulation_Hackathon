"""Calculate scores for an image dataset."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from captions import ConstantCaptions
from image_datasets import SyntheticFacesDataset
from models import (
    AestheticScoreModel,
    CLIPScoreModel,
    HPSv2ScoreModel,
    ImageRewardScoreModel,
    PickScoreModel,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_arguments() -> argparse.Namespace:
    """Define and parse the arguments from the command line.

    Returns:
        argparse.Namespace: Object containing our command line arguments.
    """
    # Define the command line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["synthetic"],
        help="Which dataset are we benchmarking on?",
    )
    parser.add_argument(
        "--caption_strategy",
        required=True,
        choices=["const"],
        help="What prompts are we using?",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the directory where images are stored.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Total batch size for our dataloader.",
    )
    parser.add_argument(
        "--const_caption",
        type=str,
        default="",
        help="If the caption is constant (i.e.caption_strategy='const'), what is it?",
    )
    parser.add_argument(
        "--disable_clip",
        action="store_true",
        default=False,
        help="Do not calculate the CLIP score.",
    )
    parser.add_argument(
        "--disable_aesthetic",
        action="store_true",
        default=False,
        help="Do not calculate the aesthetic score.",
    )
    parser.add_argument(
        "--disable_image_reward",
        action="store_true",
        default=False,
        help="Do not calculate the ImageReward score.",
    )
    parser.add_argument(
        "--disable_pick_score",
        action="store_true",
        default=False,
        help="Do not calculate the PickScore.",
    )
    parser.add_argument(
        "--disable_hps",
        action="store_true",
        default=False,
        help="Do not calculate the human preference score.",
    )

    # Load arguments in to variables
    args = parser.parse_args()
    return args


def main() -> None:
    """Calculate scores for an image dataset."""
    # Get command line arguments
    args = load_arguments()

    # Load the dataset
    if args.dataset == "synthetic":
        dataset = SyntheticFacesDataset(args.image_path)

    # Define a dataloader for our dataset
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, prefetch_factor=2
    )

    # Load our reward models
    reward_models = {}
    if args.disable_clip is False:
        reward_models["clip_score"] = CLIPScoreModel()
    if args.disable_aesthetic is False:
        reward_models["aesthetic"] = AestheticScoreModel()
    if args.disable_image_reward is False:
        reward_models["image_reward"] = ImageRewardScoreModel()
    if args.disable_pick_score is False:
        reward_models["pick_score"] = PickScoreModel()
    if args.disable_hps is False:
        reward_models["hps"] = HPSv2ScoreModel()

    # Initial our captioning strategy
    if args.caption_strategy == "const":
        captioner = ConstantCaptions(args.const_caption)

    # Define a dataframe to store our results
    results_df = pd.DataFrame()

    # Iterate over each image in the dataset
    for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        # Get the image path
        img_paths, labels = data
        print(labels)
        labels = [
            {key: tensor[i].item() for key, tensor in labels.items()}
            for i in range(len(img_paths))
        ]

        # Get the prompt for the image
        if args.caption_strategy == "const":
            prompts = captioner.get_caption(len(img_paths))
        elif args.caption_strategy == "blip":
            prompts = captioner.get_caption(img_paths)
        elif args.caption_strategy == "labels":
            prompts = captioner.get_caption(labels)

        # Define result dictionary for this image
        scores = [
            {"image_path": img_paths[i], "prompt": prompts[i]} | labels[i]
            for i in range(len(img_paths))
        ]

        # Get scores for all reward models
        for name, model in reward_models.items():
            results = model.get_scores(img_paths, prompts)
            scores = [scores[i] | {name: results[i]} for i in range(len(results))]

        # Store results in our dataframe
        scores_df = pd.DataFrame(scores)
        results_df = pd.concat([results_df, scores_df])

    # Create output folder if it doesn't exist already
    output_dir = (
        "./logs/"
        + args.dataset
        + "_"
        + args.caption_strategy
        + "_"
        + args.const_caption.replace(" ", "_")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save all results
    results_df.to_csv(os.path.join(output_dir, "scores.csv"), index=False)

    # Analyze results for each label
    for label_name, label_value in labels[0].items():
        if isinstance(label_value, float):
            # If the label is a float, we can run a simple test of the correlation
            for name, _ in reward_models.items():
                # Calculate correlation coefficient between score and labels
                score_list = results_df[name].to_list()
                label_list = results_df[label_name].to_list()
                correlation = np.corrcoef(score_list, label_list)[0, 1]

                # Create a plot showing the correlation
                plt.figure()
                plt.title(
                    "Correlation Coefficient for "
                    + label_name
                    + " : "
                    + str(np.round(correlation, 6))
                )
                plt.scatter(score_list, label_list, marker="o", s=2)
                plt.xlabel(name)
                plt.ylabel(label_name)
                plt.savefig(
                    os.path.join(output_dir, label_name + "_" + name + ".png"),
                    dpi=300,
                    bbox_inches="tight",
                )

if __name__ == "__main__":

    # Run main
    main()
