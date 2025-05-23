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

from torchvision.tv_tensors import BoundingBoxes, Image, KeyPoints, Mask, Video
from ...core import register, create


torchvision.disable_beta_transforms_warning()

ConvertBoundingBoxFormat = register()(T.ConvertBoundingBoxFormat)
RandomCrop = register()(T.RandomCrop)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomIoUCrop = register()(T.RandomIoUCrop)
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
Resize = register()(T.Resize)
SanitizeBoundingBoxes = register()(T.SanitizeBoundingBoxes)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def make_params(self, *inputs):
        return {}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt


@register()
class RandomApply(T.RandomApply):
    """
    A thin wrapper around T.RandomApply to accomodate the yaml config parser instantiating nested transform objects.
    """

    def __init__(self, transforms: list, p: float = 1.0) -> None:
        # Build the inner transforms; this is what wasn't working when trying to use T.RandomApply straight from the config.
        instances = []
        for t in transforms:
            kwargs = t.copy()
            t_type = kwargs.pop("type")
            inner_transform = create(t_type, **kwargs)
            instances.append(inner_transform)

        super().__init__(torch.nn.ModuleList(instances), p=p)


@register()
class Letterbox(T.Transform):
    """
    Letterbox transform that resizes the input to fit within a canvas of the specified size
    while preserving aspect ratio. The resized content is centered on the canvas, and the
    remaining area is padded with the given fill value and padding mode.
    
    Args:
        canvas_size (tuple[int, int]): The (height, width) of the output canvas.
        fill (int or tuple[int]): Padding fill value for image/video/mask axes.
        padding_mode (str): Padding mode passed to F.pad (e.g., "constant", "edge").

    Returns:
        The transformed input, matching the original type (e.g., PIL.Image, tv.Tensor).
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
        return padded



@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def make_params(self, *inputs):
        return {}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        tensor = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            tensor = tensor.float()
        if self.scale:
            tensor = tensor / 255.0
        return Image(tensor)

@register()
class NormalizeAnnotations(T.Resize):
    """
    Normalize TV Tensors to the range [0, 1] relative to their current canvas_size.
    Uses T.Resize to set the canvas size down to (1,1) and which will normalize any TVTensor type with supported transform kernels.
    """

    _transformed_types = (BoundingBoxes, Mask)

    def __init__(self) -> None:
        super().__init__(size=(1, 1))

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return super().transform(inpt, params)
