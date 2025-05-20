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
from torchvision.tv_tensors import wrap, BoundingBoxes, Image, KeyPoints, Mask, Video

torchvision.disable_beta_transforms_warning()


ClampBoundingBoxes = register()(T.ClampBoundingBoxes)
ColorJitter = register()(T.ColorJitter)
ConvertBoundingBoxFormat = register()(T.ConvertBoundingBoxFormat)
GaussianBlur = register()(T.GaussianBlur)
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

    def __init__(self, canvas_size, fill=0, padding_mode="constant"):
        super().__init__()
        self.canvas_size = torch.tensor(canvas_size)
        self.fill = fill
        self.padding_mode = padding_mode
    
    def make_params(self, inpt: Any) -> Dict[str, Any]:
        inpt = inpt if len(inpt) > 1 else inpt[0]
        input_h, input_w = F.get_size(inpt[0])
        canvas_h, canvas_w = self.canvas_size
        scale = min(canvas_h / input_h, canvas_w / input_w)
        content_h, content_w = int(input_h * scale), int(input_w * scale)
        pad_h, pad_w = canvas_h - content_h, canvas_w - content_w
        padding = (
            pad_w // 2,          # left
            pad_h // 2,          # top
            pad_w - pad_w // 2 , # right
            pad_h - pad_h // 2,  # bottom
        )
        return {
            "content_size": (content_h, content_w),
            "padding": padding,
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        content_size = params["content_size"]

        # Resize
        resized = F.resize(inpt, content_size)

        if isinstance(inpt, BoundingBoxes):
            # Clamp boxes to content area.
            resized = F.clamp_bounding_boxes(resized)

        # Pad
        padded = F.pad(resized, padding=params["padding"], fill=self.fill, padding_mode=self.padding_mode)
        print(f"type: {type(inpt).__name__},\tinput: {F.get_size(inpt)}, resized: {F.get_size(resized)}, padded: {F.get_size(padded)}")
        return padded


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
class NormalizeAnnotations(T.Resize):
    """
    A sneaky way of using Resize to normalize just TVTensor types, not images.
    """

    _transformed_types = (BoundingBoxes, KeyPoints,)

    def __init__(self) -> None:
        super().__init__(size=(1, 1))

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        ret = super().transform(inpt, params)
        return ret



@register()
class SanitizeBoundingBoxesWithKeyPoints(T.SanitizeBoundingBoxes):
    """
    Until SanitizeBoundingBoxes supports KeyPoints, this is the way.
    """

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # Code copied from pytorch source with KeyPoints added.
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
    """
    Just a thin wrapper around T.RandomApply since the config-driven instantiation wasn't working for passing in constructed transform types.
    """

    def __new__(cls, transform: dict, p: float = 1.0):
        # Extract and build the inner transform
        kwargs = transform.copy()
        t_type = kwargs.pop("type")
        inner_transform = create(t_type, **kwargs)

        # Wrap it using torchvision's built-in RandomApply
        return T.RandomApply(torch.nn.ModuleList([inner_transform]), p=p)
