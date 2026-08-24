"""Stage 1 preprocessing and the authoritative 8-view test-time augmentation."""

from __future__ import annotations

from typing import List

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config_binary import (
    STAGE1_IMG_SIZE,
    STAGE1_INTERPOLATION,
    STAGE1_NORM_MEAN,
    STAGE1_NORM_STD,
    STAGE1_RESIZE_SIZE,
    STAGE1_TTA_VIEWS,
)

_interp = (
    InterpolationMode.BICUBIC
    if STAGE1_INTERPOLATION == "bicubic"
    else InterpolationMode.BILINEAR
)


def build_eval_transform_8view(
    rotate_k: int = 0,
    hflip: bool = False,
) -> transforms.Compose:
    """Build one transform from the notebook’s dihedral 8-view evaluation set."""
    if rotate_k not in (0, 1, 2, 3):
        raise ValueError(f"rotate_k must be one of 0, 1, 2, 3; got {rotate_k}")

    rotate_ops = {
        0: None,
        1: Image.Transpose.ROTATE_90,
        2: Image.Transpose.ROTATE_180,
        3: Image.Transpose.ROTATE_270,
    }
    ops = []
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops.extend(
        [
            transforms.Resize(STAGE1_RESIZE_SIZE, interpolation=_interp),
            transforms.CenterCrop(STAGE1_IMG_SIZE),
        ]
    )
    if rotate_ops[rotate_k] is not None:
        ops.append(
            transforms.Lambda(
                lambda image, operation=rotate_ops[rotate_k]: image.transpose(operation)
            )
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=STAGE1_NORM_MEAN, std=STAGE1_NORM_STD),
        ]
    )
    return transforms.Compose(ops)


tta_transforms: List[transforms.Compose] = [
    build_eval_transform_8view(rotate_k=k, hflip=h)
    for h in (False, True)
    for k in (0, 1, 2, 3)
]
if len(tta_transforms) != STAGE1_TTA_VIEWS:
    raise RuntimeError(
        f"Expected {STAGE1_TTA_VIEWS} Binary TTA views, got {len(tta_transforms)}."
    )

eval_transform = tta_transforms[0]


def preprocess_image(image: Image.Image) -> list:
    """Return the eight preprocessed PIL-derived tensors in deterministic order."""
    image = image.convert("RGB")
    return [transform(image) for transform in tta_transforms]
