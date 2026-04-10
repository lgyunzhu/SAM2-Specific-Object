### Import OpenCV for window display and mouse callbacks
import cv2

## EDIT
### Load the image from disk
image = cv2.imread("/home/imyunzhu/segment-anything-2/code/elephant.png")

### Create a copy so the original stays unchanged
image_copy = image.copy()

### Define a mouse callback that captures clicks and draws a marker
def draw_circle(event, x, y, flags, param):
    ### React only to left mouse button clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        ### Draw a visible circle where the user clicked (green, filled)
        cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
        ### Print the coordinates to the console
        print(f"Point selected: ({x}, {y})")
        ### Refresh the window with the updated image
        cv2.imshow("Image", image_copy)

### Create the window
cv2.namedWindow("Image")
### Attach the callback so OpenCV reports click events
cv2.setMouseCallback("Image", draw_circle)
### Show the image initially
cv2.imshow("Image", image_copy)
### Wait until any key is pressed
cv2.waitKey(0)
### Close all OpenCV windows cleanly
cv2.destroyAllWindows()