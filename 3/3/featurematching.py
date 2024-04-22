import cv2
import numpy as np

def compute_depth(left_img, right_img, focal_length, baseline_distance):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for left image
    keypoints_left, descriptors_left = orb.detectAndCompute(left_img, None)

    # Detect keypoints and compute descriptors for right image
    keypoints_right, descriptors_right = orb.detectAndCompute(right_img, None)

    # Initialize brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_left, descriptors_right)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute disparity map
    block_size = 5
    max_disparity = 64
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disparity, blockSize=block_size)
    disparity_map = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Calculate depth using disparity map
    depth_map = (focal_length * baseline_distance) / (disparity_map + 1e-6)

    return depth_map

# Load captured images
left_img_D = cv2.imread('Left_image_d1.jpg', cv2.IMREAD_GRAYSCALE)
right_img_D = cv2.imread('Right_image_d1.jpg', cv2.IMREAD_GRAYSCALE)
left_img_DT = cv2.imread('Left_image_d2.jpg', cv2.IMREAD_GRAYSCALE)
right_img_DT = cv2.imread('Right_image_d2.jpg', cv2.IMREAD_GRAYSCALE)

# Define camera parameters
focal_length = 434  
baseline_distance = 7.5 

# Compute depth for both image pairs
depth_map_D = compute_depth(left_img_D, right_img_D, focal_length, baseline_distance)
depth_map_DT = compute_depth(left_img_DT, right_img_DT, focal_length, baseline_distance)


# Calculate and print distance D (distance to the marker)
D = np.median(depth_map_D) 
print("Distance D to the marker:", D, "cm")
# Calculate and print distance D + T (distance to the marker after the marker is moved)
# Calculate and print distance D (distance to the marker)
D_T = np.median(depth_map_DT)
print("Distance D + T to the marker:", D_T, "cm")
