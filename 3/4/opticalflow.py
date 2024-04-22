import cv2

# Load the video
video_file = 'captured_video_grayscale.mp4'  # Change this to your video file
video_capture = cv2.VideoCapture(video_file)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create a video writer object to write the result for each task
output_videos = [cv2.VideoWriter(f'3/4/optical_flow_result_task{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height)) for i in range(1, 4)]

# Initialize previous frame for each task
prev_frames = [None, None, None]

# Loop through frames
frame_count = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Task (i): Treat every previous frame as a reference frame
    if prev_frames[0] is not None:
        # Compute optical flow between consecutive frames
        flow = cv2.calcOpticalFlowFarneback(prev_frames[0], gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Plot optical flow vectors on the frame
        flow_img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        step = 16
        for y in range(0, flow_img.shape[0], step):
            for x in range(0, flow_img.shape[1], step):
                dx, dy = flow[y, x]
                cv2.arrowedLine(flow_img, (x, y), (int(x + dx), int(y + dy)), (0, 255, 0), thickness=1)

     
        output_videos[0].write(flow_img)

    prev_frames[0] = gray_frame

    # Task (ii): Treat every 11th frame as a reference frame
    if frame_count % 11 == 0:
        prev_frames[1] = gray_frame

    if prev_frames[1] is not None and frame_count > 1:
        # Compute optical flow between consecutive frames
        flow = cv2.calcOpticalFlowFarneback(prev_frames[1], gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Plot optical flow vectors on the frame
        flow_img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        step = 16
        for y in range(0, flow_img.shape[0], step):
            for x in range(0, flow_img.shape[1], step):
                dx, dy = flow[y, x]
                cv2.arrowedLine(flow_img, (x, y), (int(x + dx), int(y + dy)), (0, 255, 0), thickness=1)

    
        output_videos[1].write(flow_img)

    # Task (iii): Treat every 31st frame as a reference frame
    if frame_count % 31 == 0:
        prev_frames[2] = gray_frame

    if prev_frames[2] is not None and frame_count > 1:
        # Compute optical flow between consecutive frames
        flow = cv2.calcOpticalFlowFarneback(prev_frames[2], gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Plot optical flow vectors on the frame
        flow_img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        step = 16
        for y in range(0, flow_img.shape[0], step):
            for x in range(0, flow_img.shape[1], step):
                dx, dy = flow[y, x]
                cv2.arrowedLine(flow_img, (x, y), (int(x + dx), int(y + dy)), (0, 255, 0), thickness=1)

        output_videos[2].write(flow_img)

# Release video capture and writer for each task
for output_video in output_videos:
    output_video.release()

video_capture.release()
