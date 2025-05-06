"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import importlib.metadata
import torchvision

from packaging import version
from torch import Tensor

tv_version = importlib.metadata.version("torchvision")
has_tv_keypoints = False


if tv_version >= "0.22":
    from torchvision.tv_tensors import KeyPoints
    has_tv_keypoints = True


if tv_version >= "0.16":
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask, Video

    _boxes_keys = ["format", "canvas_size"]
elif tv_version >= "0.15.2":
    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes, BoundingBoxFormat, Image, Mask, Video
    from torchvision.transforms.v2 import SanitizeBoundingBoxes

    _boxes_keys = ["format", "spatial_size"]

else:
    raise RuntimeError("Please make sure torchvision version >= 0.15.2")



def convert_to_tv_tensor(tensor: Tensor, key: str, box_format="xyxy", spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in (
        "boxes",
        "keypoints",
        "masks",
    ), "Only support 'boxes', 'keypoints' and 'masks'"

    if key == "boxes":
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == "keypoints":
        if not has_tv_keypoints:
            raise ValueError("KeyPoint TV Tensor types are not supported in this build.")

        return KeyPoints(tensor, canvas_size=spatial_size)
    
    if key == "masks":
        return Mask(tensor)
