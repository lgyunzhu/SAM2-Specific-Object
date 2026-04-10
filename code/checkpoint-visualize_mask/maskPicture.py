### Import NumPy for array operations and reproducible randomness
import numpy as np
### Import PyTorch to select CPU or GPU
import torch
### Import Matplotlib for visualization
import matplotlib.pyplot as plt
### Import OpenCV for image loading and color conversion
import cv2

### Set a fixed seed for reproducible mask overlay colors
np.random.seed(3)

### Select computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### Print selected device to know whether running on GPU or CPU
print(f"Using device: {device}")

### Define helper function to visualize SAM2 automatically generated masks
def show_anns(anns, borders=True):
    ### If no masks, exit early
    if len(anns) == 0:
        return
    ### Sort masks by area so larger masks are drawn first
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ### Get current Matplotlib axes
    ax = plt.gca()
    ### Disable autoscale to keep overlays aligned
    ax.set_autoscale_on(False)
    
    ### Create RGBA overlay canvas with same size as first mask
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    ### Initialize with fully transparent alpha channel
    img[:, :, 3] = 0
    ### Draw each mask with random semi-transparent colors
    for ann in sorted_anns:
        ### Extract binary segmentation mask
        m = ann['segmentation']
        ### Build random RGB color with fixed alpha value
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        ### Draw selected color in mask region
        img[m] = color_mask
        ### Optional: draw borders to make mask edges easy to observe
        if borders:
            ### Import cv2 locally to match original reference style
            import cv2
            ### Find external contours of mask
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            ### Smooth contours to make borders look cleaner
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            ### Draw contour lines on RGBA overlay
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    
    ### Render overlay on image
    ax.imshow(img)

## EDIT
### Load image from disk
image = cv2.imread("code/elephant.png")
### Convert BGR to RGB for correct color display in Matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

### Create preview figure
plt.figure(figsize=(10, 10))
### Display image
plt.imshow(image)
### Hide axes for cleaner view
plt.axis('off')
### Keep show call commented to run later if needed
# plt.show()

### Import SAM2 builder
from sam2.build_sam import build_sam2
### Import automatic mask generator wrapper
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

### Select checkpoint to use
## EDIT
sam2_checkpoint = "<Path to SAM2 checkpoint, e.g. sam2.1_hiera_large.pt>"  # Download in installation section
### Select model configuration from SAM2 repository
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Part of SAM2 repository

### Build SAM2 model
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

### Create automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(sam2)

### Generate masks by running generator on image
masks = mask_generator.generate(image)

### Print how many masks were generated
print(f"Generated {len(masks)} masks")
### Print area of first mask for quick sanity check
print(f"First mask area: {masks[0]['area']}")
### Print available keys in each mask dictionary
print(masks[0].keys())

### Create large figure for mask overlay visualization
plt.figure(figsize=(20, 20))
### Draw base image first
plt.imshow(image)  # Show image first
### Overlay all generated masks
show_anns(masks)  # Image with masks
### Hide axes
plt.axis('off')
### Show final result
plt.show()