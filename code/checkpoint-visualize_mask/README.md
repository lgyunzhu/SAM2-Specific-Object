# Checkpoint Visualize Mask – SAM2 Example

This directory contains files for image and video segmentation using SAM2 (Segment Anything Model 2).

## Image Segmentation (Single Image)

- `elephant.png` – the input image of an elephant
- `getXYInImage.py` – a script to click on an image and get coordinates (used to provide a point prompt)
- `maskSpecificObject.py` – the main script that loads SAM2, takes a point prompt, and generates a mask
- `maskPicture.py` – optional script for automatic mask generation (if used)

### Usage (Image)
1. Run `getXYInImage.py` to click on the elephant and get coordinates.
2. Edit `maskSpecificObject.py` and replace the point coordinates with your click coordinates.
3. Run `maskSpecificObject.py` to segment the elephant.

## Video Segmentation (Frame Sequence)

- `maskVideo.py` – a script that processes a sequence of video frames (stored in a folder) using SAM2's video predictor.  
  **Functionality**:
  - Loads all frames (JPEG images) from a given folder.
  - Displays the first frame and allows the user to draw a bounding box around the target object (using Matplotlib's `RectangleSelector`, no Qt dependency).
  - Propagates the segmentation mask through the entire video using SAM2's memory-based architecture.
  - Saves:
    - The selected bounding box visualization (`selection_box.png`)
    - Overlay mask on the first and last frames (`first_frame_mask.png`, `last_frame_mask.png`)
    - Binary mask of the last frame (`mask_only.png`)

### Usage (Video)
1. Prepare a folder containing video frames as sequential JPEG images (e.g., `000001.jpg`, `000002.jpg`, ...).  
   **Note**: SAM2 expects numeric filenames without prefixes (e.g., `000001.jpg`). If your frames have names like `frame_000047.jpg`, rename them first using:
   ```bash
   for f in frame_*.jpg; do mv "$f" "${f#frame_}"; done

## Edit
The codes that need to be editted according to file path or names are labelled with "## EDIT"