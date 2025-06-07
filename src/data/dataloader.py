"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from functools import partial
from .transforms import RandomScale
from ..core import register


__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image"""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError("")


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size_repeat=None,
    ) -> None:
        
        super().__init__()
        self.base_size_repeat = base_size_repeat
        self.ema_restart_decay = ema_restart_decay
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000

        if base_size_repeat is None:
            self.transforms = None
        else:
            self.transforms = T.Compose([
                RandomScale(self.base_size_repeat)
            ])


    def __call__(self, items):
        if self.epoch < self.stop_epoch and self.transforms is not None:
            items = self.transforms(items)
            
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        return images, targets
