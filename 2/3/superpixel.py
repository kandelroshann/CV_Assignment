import cv2
import numpy as np

# Global variables to store clicked coordinates
clicked_x1, clicked_y1 = -1, -1
clicked_x2, clicked_y2 = -1, -1
patch_size = 16  # Adjust as needed

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global clicked_x1, clicked_y1, clicked_x2, clicked_y2

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_x1, clicked_y1 = x, y
        print("Clicked on image 1: (x1, y1) = ({}, {})".format(clicked_x1, clicked_y1))
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_x2, clicked_y2 = x, y
        print("Clicked on image 2: (x2, y2) = ({}, {})".format(clicked_x2, clicked_y2))

# Load images
image1 = cv2.imread('2/2/frames/frame_3.jpg')
image2 = cv2.imread('2/2/frames/frame_91.jpg')
# Create windows and set mouse callback
cv2.namedWindow('Image 1')
cv2.namedWindow('Image 2')
cv2.setMouseCallback('Image 1', mouse_callback)
cv2.setMouseCallback('Image 2', mouse_callback)

while True:
    # Display images
    cv2.imshow('Image 1', image1)
    cv2.imshow('Image 2', image2)

    # Check if both coordinates are selected
    if clicked_x1 != -1 and clicked_y1 != -1 and clicked_x2 != -1 and clicked_y2 != -1:
        # Extract super-pixel patches around the chosen pixels
        patch1 = image1[max(0, clicked_y1 - patch_size//2):min(image1.shape[0], clicked_y1 + patch_size//2),
                        max(0, clicked_x1 - patch_size//2):min(image1.shape[1], clicked_x1 + patch_size//2)]
        patch2 = image2[max(0, clicked_y2 - patch_size//2):min(image2.shape[0], clicked_y2 + patch_size//2),
                        max(0, clicked_x2 - patch_size//2):min(image2.shape[1], clicked_x2 + patch_size//2)]

        # Convert patches to grayscale
        gray_patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
        gray_patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

        # Display the patches
        cv2.imshow('Patch 1', gray_patch1)
        cv2.imshow('Patch 2', gray_patch2)

        # Reset clicked coordinates
        clicked_x1, clicked_y1 = -1, -1
        clicked_x2, clicked_y2 = -1, -1

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit if 'q' is pressed
        break

# Release windows
cv2.destroyAllWindows()
