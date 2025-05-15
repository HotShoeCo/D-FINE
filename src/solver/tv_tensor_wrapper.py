import torch
from types import MethodType
from torchvision.tv_tensors import BoundingBoxes, KeyPoints

def _process(obj):
    """
    Recursively wrap any dicts or lists containing 'boxes' or 'keypoints'
    in their keys with the appropriate TVTensor classes.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if "boxes" in k:
                result[k] = [BoundingBoxes(box, format="XYWH", canvas_size=(1, 1)) for box in v]
            elif "keypoints" in k:
                result[k] = [KeyPoints(pts, canvas_size=(1, 1)) for pts in v]
            else:
                result[k] = _process(v)
        return result
    elif isinstance(obj, list):
        return [_process(item) for item in obj]
    else:
        return obj

def wrap_outputs(model: torch.nn.Module) -> torch.nn.Module:
    """
    Register a forward hook so that the model's outputs are processed
    by _process without changing the forward signature or relying on kwargs.
    """
    def _hook(module, inputs, outputs):
        # outputs may be a single tensor or tuple/dict
        return _process(outputs)
    # Register the hook (it will replace outputs with processed ones)
    model.register_forward_hook(_hook)
    return model