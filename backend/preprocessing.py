"""
preprocessing.py
Stage 2 image transform pipeline - the exact resize/crop/normalize/TTA recipe used in
Multiclass_EfficientNetB3_Stage2_V1.ipynb (Section 10), built from config.py. Identical
mechanics to Stage 1's pipeline (the resize/crop/normalize logic is entirely about the image,
not the label space), kept as a separate module since Stage 2 uses its own config.py values.
"""
from typing import List

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    STAGE2_RESIZE_SIZE,
    STAGE2_IMG_SIZE,
    STAGE2_NORM_MEAN,
    STAGE2_NORM_STD,
    STAGE2_INTERPOLATION,
)

_interp = InterpolationMode.BICUBIC if STAGE2_INTERPOLATION == "bicubic" else InterpolationMode.BILINEAR


def _build_eval_transform(hflip: bool = False, vflip: bool = False) -> transforms.Compose:
    ops = []
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    if vflip:
        ops.append(transforms.RandomVerticalFlip(p=1.0))
    ops += [
        transforms.Resize(STAGE2_RESIZE_SIZE, interpolation=_interp),
        transforms.CenterCrop(STAGE2_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=STAGE2_NORM_MEAN, std=STAGE2_NORM_STD),
    ]
    return transforms.Compose(ops)


eval_transform = _build_eval_transform()
tta_transforms: List[transforms.Compose] = [
    eval_transform,
    _build_eval_transform(hflip=True),
    _build_eval_transform(vflip=True),
]


def preprocess_image(image: Image.Image) -> list:
    """Applies each of the 3 TTA-view transforms to a single PIL image (RGB) and returns a
    list of transformed tensors, one per view, ready to be stacked into a batch."""
    image = image.convert("RGB")
    return [t(image) for t in tta_transforms]
