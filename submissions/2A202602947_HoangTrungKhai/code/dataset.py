"""DeepWeeds data loading, split validation and image transforms."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    if fold not in range(5):
        raise ValueError("fold phải nằm trong 0..4")
    root = Path(labels_dir)
    frames = []
    for split in ("train", "val", "test"):
        path = root / f"{split}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if not {"Filename", "Label"}.issubset(frame.columns):
            raise ValueError(f"{path} phải có cột Filename, Label")
        frames.append(frame)
    return tuple(frames)


def check_split(train_df, val_df, test_df, images_dir: str | Path) -> dict:
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(v.Filename.astype(str)) for k, v in frames.items()}
    if any(len(names[k]) != len(frames[k]) for k in frames):
        raise ValueError("Filename bị trùng trong một split")
    overlap = {"train_val": len(names["train"] & names["val"]),
               "train_test": len(names["train"] & names["test"]),
               "val_test": len(names["val"] & names["test"])}
    if any(overlap.values()):
        raise ValueError(f"Các split bị giao nhau: {overlap}")
    union = set.union(*names.values())
    if len(union) != 17509:
        raise ValueError(f"Fold phải chứa 17.509 ảnh, hiện có {len(union)}")
    root = Path(images_dir)
    missing = [f for f in union if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh, ví dụ: {missing[:5]}")
    per_class = {k: v.Label.value_counts().reindex(range(NUM_CLASSES), fill_value=0).sort_index().astype(int).tolist()
                 for k, v in frames.items()}
    result = {"n": {k: len(v) for k, v in frames.items()}, "per_class": per_class,
              "overlap": overlap, "union": len(union), "missing": 0}
    print(result)
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    from torchvision import transforms as T
    if img_size < 32:
        raise ValueError("img_size phải >= 32")
    if not train:
        ops = [T.Resize(256), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return T.Compose(ops)
    augments = {
        "basic": [],
        "color": [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03)],
        "trivial": [T.TrivialAugmentWide()],
        "randaug": [T.RandAugment(num_ops=2, magnitude=9)],
    }
    if aug not in augments:
        raise ValueError(f"augmentation không hỗ trợ: {aug}")
    return T.Compose([T.RandomResizedCrop(img_size), T.RandomHorizontalFlip(), *augments[aug],
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row.Filename)
        with Image.open(self.images_dir / filename) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row.Label), filename


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df, images_dir, transform, batch_size: int, train: bool,
                sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    ds = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator().manual_seed(seed)
    weighted = None
    if sampler is not None:
        if not train or sampler != "balanced":
            raise ValueError("sampler chỉ hỗ trợ 'balanced' trên train")
        counts = df.Label.value_counts()
        weights = df.Label.map(lambda y: 1.0 / counts[int(y)]).to_numpy(dtype=np.float64)
        weighted = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True, generator=generator)
    return DataLoader(ds, batch_size=batch_size, shuffle=bool(train and weighted is None), sampler=weighted,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=bool(train and len(ds) >= batch_size), worker_init_fn=_seed_worker,
                      generator=generator, persistent_workers=num_workers > 0)
