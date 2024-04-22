import cv2
import numpy as np

# Load images
image1 = cv2.imread('2/2/frames/frame_3.jpg')
image2 = cv2.imread('2/2/frames/frame_91.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match keypoints
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography matrix using RANSAC
homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Compute inverse homography matrix
inverse_homography = np.linalg.inv(homography)
# Save matrices to a file
np.savetxt('2/3/homography_matrix.txt', homography)
np.savetxt('2/3/inverse_homography_matrix.txt', inverse_homography)

print("Homography Matrix:")
print(homography)
print("\nInverse Homography Matrix:")
print(inverse_homography)
