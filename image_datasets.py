"""This module contains functionality to load different datasets, which can then be used for a variety of tasks."""

import os
from typing import Any, Dict, Tuple
from torch.utils.data import Dataset
import os
from typing import Any, Dict, Tuple, List
from torch.utils.data import Dataset


class SyntheticFacesDataset(Dataset):
    """Facial Attractiveness Dataset.

    This dataset infers labels directly from image filenames.
    Filenames must contain one of the following substrings:

        - "attractive"
        - "normal"
        - "unattractive"
    
    """

    VALID_LABELS = ["unattractive", "normal", "attractive"]
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, img_dir: str) -> None:
        """Initialize the dataset.

        Args:
            img_dir: Directory containing all images. Labels are parsed from filenames.
        """
        self.img_dir = img_dir
        self.samples: List[Tuple[str, str]] = []

        for filename in os.listdir(img_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.IMAGE_EXTENSIONS:
                continue

            lower_name = filename.lower()
            label = next(
                (lbl for lbl in self.VALID_LABELS if lbl in lower_name),
                None,
            )

            if label is None:
                raise ValueError(
                    f"Could not infer label from filename: {filename}. "
                    f"Expected one of {self.VALID_LABELS}."
                )

            self.samples.append((filename, label))
        
        self.LABELS_IDX = {"unattractive" : 0.0, "normal" : 1.0, "attractive" : 2.0,}

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in directory: {img_dir}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, Dict[str, Any]]:
        """Get an item from the dataset at the specified index.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A tuple containing the image path and the attractiveness label.
        """
        filename, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)

        return img_path, {"attractiveness": self.LABELS_IDX[label]}