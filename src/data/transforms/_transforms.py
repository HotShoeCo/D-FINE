"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register, create
from .._misc import (
    BoundingBoxes,
    Image,
    KeyPoints,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)
from torchvision.tv_tensors import wrap

torchvision.disable_beta_transforms_warning()


ColorJitter = register()(T.ColorJitter)
GaussianBlur = register()(T.GaussianBlur)
Normalize = register()(T.Normalize)
RandomAffine = register()(T.RandomAffine)
RandomCrop = register()(T.RandomCrop)
RandomErasing = register()(T.RandomErasing)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomIoUCrop = register()(T.RandomIoUCrop)
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
Resize = register()(T.Resize)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs

@register()
class Letterboxed(T.Transform):
    """
    Guarantees an image fits into the desired size even if the aspect ratio isn't right.
    """
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
        KeyPoints,
    )

    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def make_params(self, inpt: Any) -> Dict[str, Any]:
        inpt = inpt if len(inpt) > 1 else inpt[0]

        # Torch vision deals sizes in (h, w) format
        h, w = F.get_size(inpt[0])
        target_w, target_h = self.size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h, pad_w = target_h - new_h, target_w - new_w
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        return {
            "new_size": (new_h, new_w),
            "padding": (pad_left, pad_top, pad_right, pad_bottom),
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:             
        resized = F.resize(inpt, params["new_size"])
        padded = F.pad(resized, padding=params["padding"], fill=self.fill, padding_mode=self.padding_mode)
        return padded


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt

@register()
class NormalizeKeyPoints(T.Transform):
    _transformed_types = (KeyPoints,)

    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, inpt: KeyPoints, params: Dict[str, Any]) -> KeyPoints:
        height, width = inpt.canvas_size
        scale = torch.tensor([width, height], device=inpt.device)
        scaled = KeyPoints(inpt / scale, canvas_size=inpt.canvas_size)
        return scaled

@register()
class SanitizeBoundingBoxesWithKeyPoints(T.SanitizeBoundingBoxes):

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = params["labels"] is not None and any(inpt is label for label in params["labels"])
        is_bounding_boxes_or_mask_or_keypoints = isinstance(
            inpt, (BoundingBoxes, Mask, KeyPoints)
        )

        if not (is_label or is_bounding_boxes_or_mask_or_keypoints):
            return inpt

        output = inpt[params["valid"]]
        if is_label:
            return output
        else:
            wrapped = wrap(output, like=inpt)
            return wrapped
        

@register()
class RandomApply:

    def __new__(cls, transform: dict, p: float = 1.0):
        # Extract and build the inner transform
        kwargs = transform.copy()
        t_type = kwargs.pop("type")
        inner_transform = create(t_type, **kwargs)

        # Wrap it using torchvision's built-in RandomApply
        return T.RandomApply(torch.nn.ModuleList([inner_transform]), p=p)
