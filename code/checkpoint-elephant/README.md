# Checkpoint Elephant – SAM2 Segmentation Example

This directory contains files used to test SAM2 (Segment Anything Model 2) for segmenting an elephant image.

- `elephant.png` – the input image of an elephant
- `getXYInImage.py` – a script to click on an image and get coordinates (used to provide a point prompt)
- `maskSpecificObject.py` – the main script that loads SAM2, takes a point prompt, and generates a mask
- `generateAutoMask.py` – optional script for automatic mask generation (if used)

## Usage
1. Run `getXYInImage.py` to click on the elephant and get coordinates.
2. Edit `maskSpecificObject.py` and replace the point coordinates with your click coordinates.
3. Run `maskSpecificObject.py` to segment the elephant.

## Edit
The codes that need to be editted according to file path or names are labelled with "## EDIT"
