import cv2
import numpy as np

# Load pre-trained YOLO model
net = cv2.dnn.readNet("3/8/yolov3.weights", "3/8/yolov3.cfg")

# Load class labels
classes = []
with open("3/8/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize variables for object tracking
prev_objects = {}
current_objects = {}

# Open video file
cap = cv2.VideoCapture("IMG_2326.MOV")


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('3/8/qr_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess input image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set input blob for the network
    net.setInput(blob)

    # Forward pass through the network
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Initialize lists to store detected objects' bounding boxes
    rects = []

    # Process detection results
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Get bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Add bounding box to the list of detected objects
                rects.append((center_x - w // 2, center_y - h // 2, center_x + w // 2, center_y + h // 2))

                # Display class label and confidence
                cv2.putText(frame, f"{classes[class_id]}: {confidence:.2f}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update object tracking based on proximity
    current_objects = {}
    for rect in rects:
        (x, y, x2, y2) = rect
        centroid = ((x + x2) // 2, (y + y2) // 2)

        matched = False
        for obj_id, prev_rect in prev_objects.items():
            prev_centroid = ((prev_rect[0] + prev_rect[2]) // 2, (prev_rect[1] + prev_rect[3]) // 2)
            if np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) < 50:
                current_objects[obj_id] = rect
                matched = True
                break

        if not matched:
            current_objects[len(current_objects)] = rect

    # Loop over the tracked objects and draw bounding boxes
    for obj_id, rect in current_objects.items():
        (x, y, x2, y2) = rect
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame into the output video
    output_video.write(frame)

    # Display result
    cv2.imshow("Object Tracking", frame)
    
    # Update previous objects for next frame
    prev_objects = current_objects.copy()
    
    # Check for key press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object, release output video object, and close all windows
cap.release()
output_video.release()
cv2.destroyAllWindows()
