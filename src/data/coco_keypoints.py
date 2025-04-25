from torchvision.tv_tensors import KeyPoints
from typing import Optional, Union, Tuple, Any
import torch


class CocoKeyPoints(KeyPoints):

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
        canvas_size: Tuple[int, int],
    ):
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[-1] < 2:
            raise ValueError(f"Expected at least 2 dims (x, y), got shape {tensor.shape}")

        # Pass only (x, y) to superclass
        base_tensor = tensor[..., :2]
        obj = super().__new__(cls, base_tensor, dtype=dtype, device=device, requires_grad=requires_grad, canvas_size=canvas_size)

        # Store remaining dims as visibility or score
        obj.visibility = tensor[..., 2:] if tensor.shape[-1] > 2 else None
        return obj