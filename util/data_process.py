import numpy as np
import torch
from torchvision.transforms import transforms


def de_normalization(image: torch.Tensor, mean=None, std=None) -> np.array:
    if std is None:
        std = [0.229, 0.224, 0.225]
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    mean = np.array(mean)
    std = np.array(std)
    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
    return denormalize(image)
