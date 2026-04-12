# YOLH Pipeline: ROS2 Bag to Imitation Learning Dataset

This repository provides a complete pipeline to convert raw ROS2 bag files (recorded during robot manipulation demonstrations) into a structured dataset for training imitation learning policies.

The pipeline consists of 6 sequential steps:

| Step | Script | Input | Output | Core Function | Model Used |
|------|--------|-------|--------|---------------|-------------|
| 00 | `00_ros2bag_process.py` | ROS2 bag directory | `raw.npz` | Extract RGB, depth, camera intrinsics | — |
| 01 | `01_hand_bbox.py` | `raw.npz` | `hand_bboxes.npz` | Hand detection + tracking | **Grounding DINO** |
| 02 | `02_mask_generation.py` | `raw.npz` + bounding boxes | `masks.npz` + `arm_bboxes.npz` | Segmentation mask generation | **SAM 2** |
| 03 | `03_hand_state.py` | `raw.npz` + bboxes + masks | `hand_state.npz` | 3D hand pose & articulation | **HaMeR** |
| 04 | `04_gripper_action.py` | `hand_state.npz` | `gripper_action.npz` | Binary gripper open/close | — |
| 05 | `05_gripper_insertion.py` | `raw.npz` + masks + action | `episodes.npz` | Point cloud + contact extraction | — |
| 06 | `06_generate_dataset.py` | All `episodes.npz` | Final dataset | Merge, normalize, format | — |

## Pipeline Overview

### Step 00: ROS2 Bag Processing
Extracts color, depth, and camera info from ROS2 bag files and saves them as compressed NumPy archives.

### Step 01: Hand Detection & Tracking
- Uses **Grounding DINO** (open-set object detector) to detect hand bounding boxes.
- Associates detections across frames to maintain temporally consistent IDs.
- Output: `hand_bboxes.npz` (T×4 array + valid flags).

### Step 02: Segmentation Mask Generation
- Uses **SAM 2** (promptable segmentation model) to generate precise hand masks.
- Optionally generates arm masks via user‑specified bounding box on the first frame.
- Output: `masks.npz` (T×H×W boolean masks).

### Step 03: Hand State Estimation
- Uses **HaMeR** (transformer‑based hand mesh recovery) to estimate 3D hand pose.
- Crops hand regions using bounding boxes and masks.
- Output: `hand_state.npz` (joint positions, rotations, etc.).

### Step 04: Gripper Action Inference
- Converts hand pose (e.g., thumb–index distance) into binary open/close commands.
- Output: `gripper_action.npz` (T×1 binary actions).

### Step 05: Point Cloud & Contact Extraction
- Reconstructs 3D point clouds from RGB‑D data.
- Filters by workspace and hand mask.
- Extracts gripper tip points and detects contact events.
- Output: `episodes.npz` (point clouds, actions, contacts).

### Step 06: Dataset Generation
- Merges all processed episodes.
- Normalizes actions and states.
- Slices into fixed‑length segments (`num_action`).
- Output: final dataset ready for training.

## Installation

```bash
# Create conda environment
conda create -n yolh python=3.10
conda activate yolh

# Install dependencies
pip install -r requirements.txt

# Install Grounding DINO, SAM 2, HaMeR (follow their official instructions)
