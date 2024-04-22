import depthai
import cv2

# Create a pipeline
pipeline = depthai.Pipeline()

# Configure the left camera node
left_cam = pipeline.create(depthai.node.MonoCamera)
left_cam.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
left_cam.setBoardSocket(depthai.CameraBoardSocket.LEFT)

# Configure the right camera node
right_cam = pipeline.create(depthai.node.MonoCamera)
right_cam.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
right_cam.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

# Create output stream for left camera
left_xout = pipeline.create(depthai.node.XLinkOut)
left_xout.setStreamName("left_video")
left_cam.out.link(left_xout.input)

# Create output stream for right camera
right_xout = pipeline.create(depthai.node.XLinkOut)
right_xout.setStreamName("right_video")
right_cam.out.link(right_xout.input)

# Start the pipeline
with depthai.Device(pipeline) as device:
    # Counter for captured images
    num_images_captured = 0
    
    # Main loop
    while num_images_captured < 5:
        # Get data from output streams of left and right cameras
        left_data = device.getOutputQueue("left_video").get()
        right_data = device.getOutputQueue("right_video").get()
        
        # Access left image data
        left_img = left_data.getFrame()
        
        # Access right image data
        right_img = right_data.getFrame()
        
        # Convert images to BGR format (OpenCV uses BGR)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
        
        # Display the captured images
        cv2.imshow("Left Image", left_img)
        cv2.imshow("Right Image", right_img)
        
        # Wait for a key press (press any key to capture the images)
        key = cv2.waitKey(1)
        
        # If the key is pressed, save the images and increment the counter
        if key != -1:
            cv2.imwrite(f"Left_Image_{num_images_captured + 1}.jpg", left_img)
            cv2.imwrite(f"Right_Image_{num_images_captured + 1}.jpg", right_img)
            num_images_captured += 1
            print(f"Images {num_images_captured} captured.")
    
    # Close OpenCV windows
    cv2.destroyAllWindows()
