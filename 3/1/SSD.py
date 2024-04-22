import cv2
import os
import random
import numpy as np

# Load the saved ROI as grayscale
roi_path = 'selected_roi1.jpg'
saved_roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

# Directory containing the dataset of frames
frames_dir = '2/2/frames'

# Get the list of frame paths
frame_paths = [os.path.join(frames_dir, frame) for frame in os.listdir(frames_dir)]

# Number of frames to randomly select
num_frames_to_select = 5  # You can adjust this number as needed

# Randomly select frames
selected_frame_paths = random.sample(frame_paths, num_frames_to_select)

# Function to calculate SSD between two images
def calculate_ssd(img1, img2):
    return np.sum((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

# Calculate SSD between saved ROI and ROIs from selected frames
for frame_path in selected_frame_paths:
    # Load the frame as grayscale
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    # Select a random region from the frame as ROI
    x1 = random.randint(0, frame.shape[1] - saved_roi.shape[1])
    y1 = random.randint(0, frame.shape[0] - saved_roi.shape[0])
    roi_from_frame = frame[y1:y1+saved_roi.shape[0], x1:x1+saved_roi.shape[1]]

    # Calculate SSD
    ssd = calculate_ssd(saved_roi, roi_from_frame)
    print("SSD between saved ROI and ROI from frame", frame_path, ":", ssd)
