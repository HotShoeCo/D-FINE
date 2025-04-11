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
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_category2name


def draw(images, labels, boxes, scores, keypoints=[], thrh=0.4, names_dict=None):
    DRAW_COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    COCO_CLASSES = mscoco_category2name.values()
    if names_dict is None:
        names_dict = {i: name for i, name in enumerate(COCO_CLASSES)}
    
    for i, im in enumerate(images):
        draw_obj = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        if i < len(keypoints):
            kpts = keypoints[i][scr > thrh]
        else:
            kpts = None

        # Boxes
        for j, b in enumerate(box):
            color = DRAW_COLORS[j % len(DRAW_COLORS)]
            draw_obj.rectangle(list(b), outline=color)
            label_id = lab[j].item()
            label_name = names_dict.get(label_id, str(label_id))
            text = f"{label_name} {round(scrs[j].item(), 2)}"
            padding = 2
            bbox = draw_obj.textbbox((0, 0), text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            # Label bg
            draw_obj.rectangle([b[0], b[1], b[0] + text_w + 2*padding, b[1] + text_h + 2*padding], fill=color)
            # Label text
            draw_obj.text((b[0] + padding, b[1] + padding), text, fill="black")

            # Keypoints
            if kpts is not None:
                draw_keypoints(draw_obj, im.size, kpts[j], color)


def draw_keypoints(draw_obj, image_size, keypoints, color, dot_radius=5):
    """Draw keypoints on the given image. Expects keypoints as a tensor of shape (1, N, num_keypoints*3) or (N, num_keypoints*3) or (N, num_keypoints, 3).

    Each detection's keypoints are processed so that if they are not already of shape (num_keypoints, 3), they are reshaped.
    Only keypoints with confidence above the threshold are drawn.
    """
    # Get image dimensions
    w, h = image_size
    num_kp = keypoints.shape[0] // 3
    kp_arr = keypoints.reshape(num_kp, 3)
    print(kp_arr)
        # # If keypoints appear to be normalized (e.g. in the range [0,1]), scale them by image width and height
        # if kp_arr.min() >= 0 and kp_arr.max() <= 1.1:
        #     kp_arr[:, 0] = kp_arr[:, 0] * w
        #     kp_arr[:, 1] = kp_arr[:, 1] * h
        
    for x, y, conf in kp_arr:
        print(x, y, conf)
        draw_obj.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=color)


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

    draw([im_pil], labels, boxes, scores, keypoints)
    im_pil.save("torch_results.jpg")


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
        draw([frame_pil], labels, boxes, scores)
        if keypoints is not None:
            draw_keypoints(frame_pil, keypoints)

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
