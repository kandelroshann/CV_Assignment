import cv2

# Load the pre-trained QR code detector
detector = cv2.QRCodeDetector()

# Open the video file
video_path = "IMG_2325.MOV"  # Replace this with the path to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties (width, height, and fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = "3/8/qr_output_video.mp4"  # Path to output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect QR codes in the frame
    data, bbox, _ = detector.detectAndDecode(frame)
    
    # If a QR code is detected, draw only the bounding box
    if bbox is not None:
        bbox = bbox[0].astype(int)  # Convert bbox points to integers
        for i in range(len(bbox)):
            cv2.line(frame, tuple(bbox[i]), tuple(bbox[(i+1) % len(bbox)]), color=(0, 255, 0), thickness=2)
    
    # Write the frame to the output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow('QR Code Detector', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the output video
cap.release()
out.release()
cv2.destroyAllWindows()
