"""
PyTorch Dataset and DataLoader for Sentinel-2 Indian Satellite Deforestation Patches.
"""

import os
from typing import Callable, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SentinelDeforestationDataset(Dataset):
    """
    Dataset representing real Sentinel-2 satellite patches over India
    labeled for deforestation monitoring.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        use_metadata: bool = True,
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.classes = ["intact_forest", "deforested"]

        csv_path = os.path.join(data_dir, "metadata.csv")
        if use_metadata and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            split_df = df[df["split"] == split].reset_index(drop=True)
            for _, row in split_df.iterrows():
                img_path = os.path.join(data_dir, row["file_path"])
                if os.path.exists(img_path):
                    self.samples.append((img_path, int(row["label"]), row["class_name"]))
        else:
            # Fallback to directory walking
            split_dir = os.path.join(data_dir, split)
            for label, cls_name in enumerate(self.classes):
                cls_dir = os.path.join(split_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        self.samples.append(
                            (os.path.join(cls_dir, fname), label, cls_name)
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label, cls_name = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        return tensor, label, cls_name


def get_satellite_transforms(img_size: int = 128) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training and validation/test torchvision transform pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, val, and test DataLoaders for satellite deforestation patches."""
    train_tf, eval_tf = get_satellite_transforms(img_size)

    train_ds = SentinelDeforestationDataset(data_dir, split="train", transform=train_tf)
    val_ds = SentinelDeforestationDataset(data_dir, split="val", transform=eval_tf)
    test_ds = SentinelDeforestationDataset(data_dir, split="test", transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
