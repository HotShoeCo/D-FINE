"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import PIL
import PIL.Image
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from torchvision.tv_tensors import BoundingBoxes, Image, KeyPoints, Mask, Video, wrap
from typing import Any, Dict
from ...core import register, create


ClampBoundingBoxes = register()(T.ClampBoundingBoxes)
ColorJitter = register()(T.ColorJitter)
ConvertBoundingBoxFormat = register()(T.ConvertBoundingBoxFormat)
GaussianBlur = register()(T.GaussianBlur)
RandomAffine = register()(T.RandomAffine)
RandomCrop = register()(T.RandomCrop)
RandomErasing = register()(T.RandomErasing)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
Resize = register()(T.Resize)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    """
    A thin wrapper around T.RandomIoUCrop to add a p param.
    """

    _transformed_types = (
        Image,
        Video,
        Mask,
        BoundingBoxes,
        KeyPoints,
    )


    def __init__(self, p: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p = p

    def make_params(self, inpts: Any) -> Dict[str, Any]:
        # Parent class will return empty params if there's no valid IoU crop to be made.
        parent_params = super().make_params(inpts)
        r = random.random()

        # Simulate no IoU if the probably test failed.
        params = {} if r > self.p else parent_params
        return params

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params:
            return inpt

        return super().transform(inpt, params)


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
        The transformed input, matching the original type.
    """
    _transformed_types = (
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

    def make_params(self, inpts: Any) -> Dict[str, Any]:
        img = next(x for x in inpts if isinstance(x, Image))
        input_h, input_w = F.get_size(img)
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
    A Transform that undoes a Letterbox(…) operation given the original image size (h,w).

    It recomputes the same scale & padding that Letterbox would have used, then subtracts the padding and divides
    by that scale. Finally, it clamps so no coordinate exceeds the original image's boundaries.
    """

    _transformed_types = (
        Image,
        Video,
        Mask,
        BoundingBoxes,
        KeyPoints,
    )


    def __call__(self, *inpts: Any):
        self.orig_canvas_size = inpts[0]["orig_canvas_size"]
        return super().__call__(inpts)

    def make_params(self, inpts: Any) -> Dict[str, Any]:
        img = next(x for x in inpts if isinstance(x, Image))
        lb_canvas_h, lb_canvas_w = F.get_size(img)
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
class OriginalSize(T.Transform):
    """
    Resizes targets to the original input size passed in via `orig_canvas_size` in the caller's dict values.
    This size is in (h,w) format following torchvision size conventions.
    """

    _transformed_types = (
        Image,
        Video,
        Mask,
        BoundingBoxes,
        KeyPoints,
    )


    def __call__(self, *inpt: Any):
        self.orig_canvas_size = inpt[0]["orig_canvas_size"]
        return super().__call__(inpt)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        resized = F.resize(inpt, size=self.orig_canvas_size)
        return resized


@register()
class ToImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype=torch.float32, scale=True) -> None:
        super().__init__()
        if isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        image = F.to_image(inpt)
        image = F.to_dtype(image, dtype=self.dtype, scale=self.scale)
        return image

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
class Denormalize(T.Transform):
    """
    Denormalizes TV Tensors to the size of the first image in the inputs to make_params.
    """

    _transformed_types = (BoundingBoxes, Image, KeyPoints, Mask)

    def make_params(self, inpts: Any) -> Dict[str, Any]:
        img = next(x for x in inpts if isinstance(x, Image))
        return {
            "canvas_size": F.get_size(img)
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, Image):
            # Image was just passed in to get canvas_size for the rest of the annotations.
            return inpt

        canvas_size = params["canvas_size"]
        resized = F.resize(inpt, canvas_size)
        return resized


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
class EncodeInvisibleKeyPoints(T.Transform):
    """
    Some annotations styles will place invisible keypoints at 0,0 in the image.
    This will set those values to (NaN, NaN) so further transforms ignore them.
    This must be followed up at the end of the transformation pipeline by DecodeInvisibleKeyPoints.
    """

    _transformed_types = (KeyPoints,)


    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        coords = inpt.clone()
        mask = (coords[..., 0] == 0) & (coords[..., 1] == 0)
        coords[mask] = float("nan")
        return coords


@register()
class DecodeInvisibleKeyPoints(T.Transform):
    """
    After all the geometric ops, find any NaN in KeyPoints and force it back to 0.
    This returns invisible keypoints to (0,0) on the final canvas.
    """

    _transformed_types = (KeyPoints,)


    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        coords = inpt.clone()
        nan_mask = torch.isnan(coords)
        coords[nan_mask] = 0.0
        return coords


@register()
class RandomScale(T.Transform):
    """
    Choose a random square multiplier around the base size and resize both images and annotations accordingly.
    """

    _transformed_types = (BoundingBoxes, Image, KeyPoints, Mask)


    def __init__(self, base_size_repeat: int = 3):
        super().__init__()
        self.base_size_repeat = base_size_repeat

    def _generate_scales(self, base_size, base_size_repeat):
        scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
        scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
        scales += [base_size] * base_size_repeat
        scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
        return scales

    def make_params(self, inpts: Any) -> Dict[str, Any]:
        # This transform is called with an entire batch of images and their annotations. The entire batch needs to be
        # resized the same, so the scaling will be based off the size of the first image we encounter.
        img = next(x for x in inpts if isinstance(x, Image))
        h, w = F.get_size(img)
        scales = self._generate_scales(h, self.base_size_repeat)
        new_h = random.choice(scales)
        scale = new_h / float(h)
        new_w = int(round(w * scale))
        return {"size": (new_h, new_w)}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        size = params["size"]
        resized = F.resize(inpt, size=size)
        return resized
