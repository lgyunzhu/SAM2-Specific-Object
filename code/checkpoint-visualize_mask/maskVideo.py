import os
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_ROOT = os.path.join(PROJECT_ROOT, "sam2")
sys.path.insert(0, SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor



# Visualization Helpers

def save_box_visualization(frame, bbox, save_path):

    x1,y1,x2,y2 = bbox.astype(int)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(frame)

    rect = plt.Rectangle(
        (x1,y1),
        x2-x1,
        y2-y1,
        edgecolor="lime",
        facecolor="none",
        linewidth=3
    )

    ax.add_patch(rect)
    ax.set_title("User Selected Bounding Box")
    ax.axis("off")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_mask_overlay(frame, mask, save_path, title):

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(frame)

    overlay = np.zeros((*mask.shape,4))
    overlay[mask>0] = [0,0.7,1,0.6]

    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_mask_only(mask, save_path):

    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_img)


# User Bounding Box Selection

def get_bbox_from_user(rgb_frame):

    bbox = {}

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(rgb_frame)
    ax.set_title("Drag a box and close the window")
    ax.axis("off")

    def onselect(eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        bbox["coords"] = [
            min(x1,x2),
            min(y1,y2),
            max(x1,x2),
            max(y1,y2)
        ]

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        interactive=True
    )

    plt.show()

    if "coords" not in bbox:
        raise ValueError("No bounding box selected")

    return np.array(bbox["coords"], dtype=np.float32)


# Load Frames from Folder

def load_frames_from_folder(frame_dir):

    frame_files = sorted(os.listdir(frame_dir))

    frames = []

    for f in frame_files:

        path = os.path.join(frame_dir,f)

        img = cv2.imread(path)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        frames.append(img)

    frames = np.array(frames)

    print("Loaded frames:",frames.shape)

    return frames


# Main Mask Generation

def generate_masks_from_folder(
    frame_dir,
    output_dir,
    sam2_checkpoint,
    sam2_config,
    device="auto"
):

    os.makedirs(output_dir,exist_ok=True)

    if device=="auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rgb_frames = load_frames_from_folder(frame_dir)

    num_frames = len(rgb_frames)

    print("Total frames:",num_frames)

    predictor = build_sam2_video_predictor(
        sam2_config,
        sam2_checkpoint,
        device=device
    )

    inference_state = predictor.init_state(video_path=frame_dir)


    # USER BOX SELECTION
    bbox = get_bbox_from_user(rgb_frames[0])

    print("Selected bbox:",bbox)

    save_box_visualization(
        rgb_frames[0],
        bbox,
        os.path.join(output_dir,"selection_box.png")
    )


    # Add Prompt
    _, obj_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=bbox
    )

    # Propagate Masks
    H,W,_ = rgb_frames[0].shape

    masks = np.zeros((num_frames,H,W),dtype=np.bool_)

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):

        mask = (mask_logits[0]>0).cpu().numpy().squeeze()

        masks[frame_idx] = mask

    # Save Visualizations
    save_mask_overlay(
        rgb_frames[0],
        masks[0],
        os.path.join(output_dir,"first_frame_mask.png"),
        "First Frame Mask"
    )

    save_mask_overlay(
        rgb_frames[-1],
        masks[-1],
        os.path.join(output_dir,"last_frame_mask.png"),
        "Last Frame Mask"
    )

    save_mask_only(
        masks[-1],
        os.path.join(output_dir,"mask_only.png")
    )

    print("Saved demo images to:",output_dir)

    return masks


# Example Run

if __name__=="__main__":
    ## EDIT
    frame_dir = "<Folder containing video frames as images>"
    ## EDIT
    output_dir = "./demo_outputs"
    ## EDIT
    sam2_checkpoint = "<Path to SAM2 checkpoint, e.g. sam2.1_hiera_large.pt>"

    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    generate_masks_from_folder(
        frame_dir,
        output_dir,
        sam2_checkpoint,
        sam2_config
    )

