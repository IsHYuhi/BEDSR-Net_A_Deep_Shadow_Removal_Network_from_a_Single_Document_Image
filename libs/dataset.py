from logging import getLogger
from typing import Any, Dict, Optional

import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader, Dataset

# from torchvision import transforms
import albumentations as A

from .dataset_csv import DATASET_CSVS

__all__ = ["get_dataloader"]

logger = getLogger(__name__)


def get_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[A.Compose] = None,
) -> DataLoader:
    if dataset_name not in DATASET_CSVS:
        message = f"dataset_name should be selected from {list(DATASET_CSVS.keys())}."
        logger.error(message)
        raise ValueError(message)

    if split not in ["train", "val", "test"]:
        message = "split should be selected from ['train', 'val', 'test']."
        logger.error(message)
        raise ValueError(message)

    logger.info(f"Dataset: {dataset_name}\tSplit: {split}\tBatch size: {batch_size}.")

    csv_file = getattr(DATASET_CSVS[dataset_name], split)

    data = BackGroundDataset(csv_file, transform=transform)
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


class BackGroundDataset(Dataset):
    def __init__(self, csv_file: str, transform: Optional[A.Compose] = None) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        self.transform = transform

        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.df.iloc[idx]["img"]

        img = cv2.imread(img_path)

        if self.transform is not None:
            img = self.transform(image=img)

        bgr = torch.Tensor([self.df.iloc[idx]["B"], self.df.iloc[idx]["G"], self.df.iloc[idx]["R"]])/255

        sample = {"img": img["image"], "bgr": bgr, "img_path": img_path}

        return sample


class ShadowDocumentDataset(Dataset):
    def __init__(self, csv_file: str, transform: Optional[A.Compose] = None) -> None:
        super().__init__()

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError("csv file not found.") as e:
            logger.exception(f"{e}")

        self.transform = transform

        logger.info(f"the number of samples: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.df.iloc[idx]["img"]
        gt_path = self.df.iloc[idx]["gt"]

        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)

        if self.transform is not None:
            res = self.transform(image=img, mask=gt)
            img = res["image"]
            gt = res["mask"]

        sample = {"img": img, "gt": gt, "img_path": img_path}

        return sample
