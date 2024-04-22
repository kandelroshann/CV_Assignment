import cv2
import numpy as np
# Load images
image1 = cv2.imread('2/2/frames/frame_3.jpg')
image2 = cv2.imread('2/2/frames/frame_91.jpg')

# Create SIFT object
sift = cv2.SIFT_create()

# Detect and compute SIFT features
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

# Draw lines representing keypoint orientation on images
image1_with_orientation = cv2.drawKeypoints(image1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_with_orientation = cv2.drawKeypoints(image2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Compute SSD between descriptors of the first keypoint
ssd = np.sum((des1[0] - des2[0])**2)
print("SSD between descriptors of the first keypoint:", ssd)

# Display images with keypoints and orientation lines
cv2.imwrite('2/3/image1_with_orientation.jpg', image1_with_orientation)
cv2.imwrite('2/3/image2_with_orientation.jpg', image2_with_orientation)
cv2.imshow('Image 1 with SIFT keypoints and orientation', image1_with_orientation)
cv2.imshow('Image 2 with SIFT keypoints and orientation', image2_with_orientation)
cv2.waitKey(0)
cv2.destroyAllWindows()
