### Import NumPy for arrays and point formatting
import numpy as np
### Import PyTorch for device selection
import torch
### Import Matplotlib for visualization
import matplotlib.pyplot as plt
### Import OpenCV for reading and converting images
import cv2

### Set a fixed seed for reproducible visualization colors
np.random.seed(3)

### Select the device for computation
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
### Print the chosen device
print(f"Using device: {device}")


### Define a helper that overlays multiple masks
def show_anns(anns, borders=True):
    ### Exit if there are no masks
    if len(anns) == 0:
        return
    ### Sort by area so larger masks draw first
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ### Get current axes
    ax = plt.gca()
    ### Keep axes stable
    ax.set_autoscale_on(False)
    
    ### Create an empty RGBA overlay
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    ### Make it transparent by default
    img[:, :, 3] = 0
    ### Paint each mask
    for ann in sorted_anns:
        ### Extract segmentation
        m = ann['segmentation']
        ### Assign a random semi-transparent color
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        ### Apply color to mask region
        img[m] = color_mask
        ### Optionally draw borders for clarity
        if borders:
            ### Find the mask contour edges
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            ### Smooth edges
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            ### Draw edges
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ### Render overlay
    ax.imshow(img)


### Define a helper that overlays a single mask
def show_mask(mask, ax, random_color=False, borders=True):
    ### Choose random or fixed color
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    ### Read mask height and width
    h, w = mask.shape[-2:]
    ### Convert to uint8 for contour ops
    mask = mask.astype(np.uint8)
    ### Build RGBA overlay
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ### Optionally draw borders
    if borders:
        ### Find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ### Smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        ### Draw contours on the overlay
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ### Show overlay
    ax.imshow(mask_image)


### Define a helper to show positive and negative points
def show_points(coords, labels, ax, marker_size=375):
    ### Separate positive points
    pos_points = coords[labels==1]
    ### Separate negative points
    neg_points = coords[labels==0]
    ### Draw positive points
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ### Draw negative points
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)


### Define a helper to draw a bounding box
def show_box(box, ax):
    ### Extract top-left
    x0, y0 = box[0], box[1]
    ### Compute width and height
    w, h = box[2] - box[0], box[3] - box[1]
    ### Draw a rectangle patch
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                               facecolor=(0, 0, 0, 0), lw=2))


### Define a helper to display masks and their scores
def show_masks(image, masks, scores, point_coords=None, box_coords=None, 
               input_labels=None, borders=True):
    ### Loop through mask candidates
    for i, (mask, score) in enumerate(zip(masks, scores)):
        ### Create a figure for each mask
        plt.figure(figsize=(10, 10))
        ### Show the image
        plt.imshow(image)
        ### Overlay the mask
        show_mask(mask, plt.gca(), borders=borders)
        ### If points are provided, overlay them too
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        ### If a box is provided, overlay it too
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        ### Add a title if multiple scores exist
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        ### Hide axes
        plt.axis('off')
        ### Show plot
        plt.show()

## EDIT
### Load the image from disk
image = cv2.imread("/home/imyunzhu/segment-anything-2/code/elephant.png")
### Convert to RGB for Matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

### Preview the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
# plt.show()

### Import SAM2 model builder
from sam2.build_sam import build_sam2
### Import the SAM2 image predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

### Choose a checkpoint
sam2_checkpoint = "/home/imyunzhu/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"  # download it in the install part
### Choose the model configuration
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # part of the SAM2 repo

### Build the SAM2 model
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
### Create the predictor wrapper
predictor = SAM2ImagePredictor(sam2_model)

### Compute and store the image embedding
predictor.set_image(image)

### Define a single positive point for the small elephant
## EDIT
input_point = np.array([[1000, 650]])
### Label 1 means positive foreground point
input_label = np.array([1])  # 1 for positive point, 0 for negative point

### Visualize the chosen point
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

### Print embedding shapes to confirm the image was embedded
print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

### Predict masks from point prompts
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,)  # Set to True to get multiple masks

### Print how many masks were produced
print(f"Generated {len(masks)} masks")
### Print mask tensor shape
print(masks.shape)

### Display masks with scores
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)