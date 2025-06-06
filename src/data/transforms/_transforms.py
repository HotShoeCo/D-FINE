"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from torchvision.tv_tensors import BoundingBoxes, Image, KeyPoints, Mask, Video, wrap
from typing import Any, Dict, Tuple
from ...core import register, create


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
class UnLetterbox(T.Transform):
    """
    A Transform that undoes a Letterbox(…) operation on BoundingBoxes or KeyPoints,
    given the original image size (h,w).

    It recomputes the same scale & padding that Letterbox would have used,
    then subtracts the padding and divides by that scale. Finally, it clamps
    so no coordinate exceeds the original image's boundaries.

    Usage:
       unletter = UnLetterbox(orig_size=(orig_h,orig_w), canvas_size=(canvas_h,canvas_w))
       boxes_640 = ...        # a BoundingBoxes in the 640x640 frame
       boxes_orig = unletter.transform(boxes_640, params=None)
       # output boxes_orig has shape [N,4], in the original pixel frame.

       kp_640 = ...           # KeyPoints in the 640x640 frame
       kp_orig = unletter.transform(kp_640, params=None)
    """

    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
        KeyPoints,
    )


    def __call__(self, *inpt: Any):
        self.orig_canvas_size = inpt[0]["orig_canvas_size"]
        return super().__call__(inpt)

    def make_params(self, flattened_inputs: Any) -> Dict[str, Any]:
        input_img = flattened_inputs[0]
        lb_canvas_h, lb_canvas_w = F.get_size(input_img)
        orig_content_h, orig_content_w = self.orig_canvas_size

        # Recompute scale and padding exactly like Letterbox.make_params.
        scale = min(lb_canvas_h / orig_content_h, lb_canvas_w / orig_content_w)
        lb_content_h = int(orig_content_h * scale)
        lb_content_w = int(orig_content_w * scale)

        pad_h_total = lb_canvas_h - lb_content_h
        pad_w_total = lb_canvas_w - lb_content_w

        pad_top = pad_h_total // 2
        pad_left = pad_w_total // 2
        # pad_bottom, pad_right are unnecessary.

        return {
            "content_size": (lb_content_h, lb_content_w),
            "pad_left": pad_left,
            "pad_top": pad_top,
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        content_h, content_w = params["content_size"]
        pad_left = params["pad_left"]
        pad_top = params["pad_top"]
        cropped = F.crop(inpt, pad_top, pad_left, content_h, content_w)
        resized = F.resize(cropped, size=self.orig_canvas_size)
        return resized


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

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

    _transformed_types = (BoundingBoxes, KeyPoints, Mask)

    def __init__(self) -> None:
        super().__init__(size=(1, 1))

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return super().transform(inpt, params)


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
