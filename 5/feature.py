import cv2
import numpy as np

# Load the box image
box_image = cv2.imread('3/5/object.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the box image is loaded successfully
if box_image is None:
    print("Error: Could not load the box image.")
    exit()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and extract descriptors for the box image
box_keypoints, box_descriptors = sift.detectAndCompute(box_image, None)

# Load scene images
paths = ["2/2/frames/frame_10.jpg","2/2/frames/frame_29.jpg","2/2/frames/frame_35.jpg","2/2/frames/frame_39.jpg","2/2/frames/frame_2.jpg","2/2/frames/frame_7.jpg","2/2/frames/frame_15.jpg","2/2/frames/frame_19.jpg","2/2/frames/frame_29.jpg"]

# Initialize matcher outside the loop
matcher = cv2.BFMatcher()

# Loop through each scene image
for path in paths:
    scene_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Check if the scene image is loaded successfully
    if scene_image is None:
        print("Error: Could not load the scene image", path)
        continue

    # Detect keypoints and extract descriptors for the scene image
    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_image, None)

    # Match features between box and scene images
    matches = matcher.knnMatch(box_descriptors, scene_descriptors, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    ratio_threshold = 0.75
    for best_match, second_best_match in matches:
        if best_match.distance < ratio_threshold * second_best_match.distance:
            good_matches.append(best_match)

    # Extract matched keypoints
    matched_box_points = np.float32([box_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    matched_scene_points = np.float32([scene_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # Find affine transformation between matched points
    M, _ = cv2.estimateAffine2D(matched_box_points, matched_scene_points)

    # Transform box corners to scene perspective
    box_corners = np.float32([[0, 0], [box_image.shape[1], 0], [box_image.shape[1], box_image.shape[0]], [0, box_image.shape[0]]]).reshape(-1, 1, 2)
    new_box_corners = cv2.transform(box_corners, M)

    # Draw box polygon on the scene image
    scene_with_box = cv2.cvtColor(scene_image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(scene_with_box, [np.int32(new_box_corners)], isClosed=True, color=(0, 255, 255), thickness=2)

    # Save the scene image with detected box
    output_path = "3/5/detected_" + path.split('/')[-1]  # Example output path
    cv2.imwrite(output_path, scene_with_box)

    # Display scene image with detected box
    cv2.imshow('Detected Object', scene_with_box)
    cv2.waitKey(1000)  # Add a delay of 1 second between each image

# Close all windows at the end
cv2.destroyAllWindows()
