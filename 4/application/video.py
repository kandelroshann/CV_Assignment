import depthai
import cv2
import time

# Create a pipeline
pipeline = depthai.Pipeline()

# Configure the camera node
cam = pipeline.create(depthai.node.ColorCamera)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create output stream
xout = pipeline.create(depthai.node.XLinkOut)
xout.setStreamName("video")
cam.video.link(xout.input)

# Start the pipeline
with depthai.Device(pipeline) as device:
    # Flag to indicate whether to start recording
    start_recording = False
    
    # Start time for recording
    start_time = None
    
    # Get camera frame rate
    fps = 13

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("4/captured_video_rgb.mp4", fourcc, fps, (1920, 1080), isColor=True)  # Set isColor=True for RGB
    
    # Main loop
    while True:
        # Get data from output stream
        data = device.getOutputQueue("video").get()
        
        # Access image data
        img = data.getCvFrame()
        
        # Display the captured image
        cv2.imshow("Image", img)
        
        # Wait for a key press
        key = cv2.waitKey(1)
        
        # If the key is pressed, start recording
        if key != -1 and not start_recording:
            start_recording = True
            start_time = time.time()
            print("Recording started.")
        
        # If recording has started, record video frames until the user presses a specific key ('q' for example)
        if start_recording:
            # Write video frame
            out.write(img)
            
            # Check if the specific key is pressed
            if key == ord('q'):
                print("Recording stopped.")
                break
    
    # Release video writer
    out.release()
    
    # Close OpenCV windows
    cv2.destroyAllWindows()
