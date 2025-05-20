import torch
import torchvision.transforms.v2.functional as F

from torchvision.tv_tensors import BoundingBoxes, KeyPoints

  
def wrap(results, old_box_format, old_canvas_size, new_box_format=None, new_canvas_size=None):
    """
    Recursively wrap any dicts or lists containing 'boxes' or 'keypoints'
    in their keys with the appropriate TVTensor classes. BoundingBoxes are reformatted to CXCYWH.
    BoundingBoxes and KeyPoints are denormalized according to `sample_size`.
    """
    if isinstance(results, dict):
        result = {}
        for k, v in results.items():
            if "boxes" in k:
                # Raw outputs are in normalized space in XYWH format, parse as such then convert to CXCYWH.
                # Parse raw data as-is.
                b = BoundingBoxes(v, format=old_box_format, canvas_size=old_canvas_size)
                
                # Resize/reformat as desired.
                if new_box_format:
                    b = F.convert_bounding_box_format(b, new_format=new_box_format)
                
                if new_canvas_size:
                    b = F.resize(b, new_canvas_size)

                result[k] = b

                # Debug any invalid boxes in this list
                bb_xyxy = F.convert_bounding_box_format(b, new_format="XYXY")
                coords = bb_xyxy.data
                invalid_mask = (coords[:, 2] < coords[:, 0]) | (coords[:, 3] < coords[:, 1])
                if invalid_mask.any():
                    print(f"[DEBUG] _process invalid boxes at indices:", invalid_mask.nonzero(as_tuple=True)[0].tolist())
                    print(coords[invalid_mask])

            elif "keypoints" in k:
                # Parse raw data.
                kp = KeyPoints(v, canvas_size=old_canvas_size)

                # Resize as needed.
                if new_canvas_size:
                    kp = F.resize(kp, new_canvas_size)

                result[k] = kp
            
            else:
                result[k] = wrap(v, old_box_format, old_canvas_size, new_box_format, new_canvas_size)

        return result
    elif isinstance(results, list):
        return [wrap(item, old_box_format, old_canvas_size, new_box_format, new_canvas_size) for item in results]
    else:
        return results
    

def unwrap(results, new_box_format=None):
    """
    Recursively unwrap any dicts or lists containing 'boxes' or 'keypoints' from their TVTensor types down to normal torch.tensor.
    if bbox_format is specified, conversion will take place. Otherwise the data is unaltered.
    """
    if isinstance(results, dict):
        result = {}
        for k, v in results.items():
            if "boxes" in k:
                if new_box_format:
                    # Ensure desired format.
                    v = [
                        F.convert_bounding_box_format(b, new_format=new_box_format)
                        for b in v
                    ]

                result[k] = torch.stack(v, dim=0)
            elif "keypoints" in k:
                result[k] = torch.stack(v)
            else:
                result[k] = unwrap(v, new_box_format)

        return result
    elif isinstance(results, list):
        return [unwrap(item, new_box_format) for item in results]
    else:
        return results