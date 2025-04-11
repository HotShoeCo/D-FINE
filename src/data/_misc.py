"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import importlib.metadata

from torch import Tensor

tv_ver = importlib.metadata.version("torchvision")

if tv_ver >= "0.16":
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask, Video

    _boxes_keys = ["format", "canvas_size"]
elif tv_ver >= "0.15.2":
    import torchvision
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
        "masks",
    ), "Only support 'boxes' and 'masks'"

    if key == "boxes":
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == "masks":
        return Mask(tensor)
