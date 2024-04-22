import cv2

# Function to detect objects
def detect_objects(frame):
    # Convert the frame from BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to segment the image
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store detected objects
    detected_objects = []
    
    # Loop over the contours
    for contour in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ignore small contours (noise)
        if cv2.contourArea(contour) > 1200:
            # Add the bounding box coordinates to the list of detected objects
            detected_objects.append((x, y, w, h))
    
    return detected_objects

# Open the video file
cap = cv2.VideoCapture('4/captured_video_rgb.mp4')

# Iterate through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects in the current frame
    detected_objects = detect_objects(frame)
    
    # Draw bounding boxes around detected objects
    for (x, y, w, h) in detected_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame with detected objects
    cv2.imshow('Frame', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
