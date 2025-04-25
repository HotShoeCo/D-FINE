"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import draw_bounding_boxes, draw_keypoints
import torchvision.transforms.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_category2name


def draw_keypoints_on_image(image_tensor, keypoints_tensor, threshold=0.5):
    """
    Draw keypoints using torchvision's draw_keypoints correctly.
    image_tensor: Tensor (3, H, W)
    keypoints_tensor: (N, num_keypoints, 3)
    """
    xy = keypoints_tensor[..., :2]  # (N, K, 2)
    conf = keypoints_tensor[..., 2]  # (N, K)

    visibility = conf > threshold  # (N, K)

    result = draw_keypoints(
        image_tensor,
        xy,
        visibility=visibility,
        radius=5,
        colors="red",
    )
    return result


def draw(images, labels, boxes, scores, keypoints=[], thrh=0.4, names_dict=None):
    COCO_CLASSES = mscoco_category2name.values()
    if names_dict is None:
        names_dict = {i: name for i, name in enumerate(COCO_CLASSES)}

    for i, im in enumerate(images):
        # Convert image to tensor and ensure uint8 type for drawing
        if isinstance(im, Image.Image):
            image_tensor = F.to_tensor(im)
            image_tensor = (image_tensor * 255).to(torch.uint8)
        else:
            image_tensor = im
            if image_tensor.dtype != torch.uint8:
                image_tensor = (image_tensor * 255).to(torch.uint8)

        scr = scores[i]
        keep = scr > thrh

        lab = labels[i][keep]
        box = boxes[i][keep]
        scrs = scr[keep]

        has_keypoints = keypoints is not None and i < len(keypoints)
        kpts = keypoints[i][keep] if has_keypoints else None

        labels_list = []
        for j in range(len(lab)):
            label_id = lab[j].item()
            label_name = names_dict.get(label_id, str(label_id))
            text = f"{label_name} {round(scrs[j].item(), 2)}"
            labels_list.append(text)

        if len(box) > 0:
            image_tensor = draw_bounding_boxes(
                image_tensor,
                box,
                labels=labels_list,
                width=2,
                font_size=12,
            )

        if kpts is not None and len(kpts) > 0:
            image_tensor = draw_keypoints_on_image(image_tensor, kpts)

        images[i] = F.to_pil_image(image_tensor)


def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    if len(output) == 4:
        labels, boxes, scores, keypoints = output
    else:
        labels, boxes, scores = output
        keypoints = []

    images = [im_pil]
    draw(images, labels, boxes, scores, keypoints)
    images[0].save("torch_results.jpg")


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("torch_results.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        if len(output) == 4:
            labels, boxes, scores, keypoints = output
        else:
            labels, boxes, scores = output
            keypoints = None

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores, keypoints if keypoints is not None else [])
        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(model, device, file_path)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
