"""Pretrained ResNet18 embeddings for Gardens Point visual place recognition.

Image classification logits are discarded; use the global-average-pooled
512-dimensional feature vector, then L2-normalize for cosine retrieval.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            return self.transform(image)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(device: torch.device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms()


@torch.inference_mode()
def embed_images(paths: list[Path], model, transform, device: torch.device, batch_size: int = 16) -> np.ndarray:
    loader = DataLoader(ImageDataset(paths, transform), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    features = []
    for batch in loader:
        batch = batch.to(device)
        features.append(model(batch).float().cpu())
    vectors = torch.cat(features, dim=0)
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    return vectors.numpy().astype('float32')
