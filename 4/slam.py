import cv2
import numpy as np
from depthai import Device, Pipeline, StereoDepth, Point2f
from depthai import ColorCameraProperties
from flask import Flask, Response, render_template


def run_slam():
    # Initialize DepthAI device
    with Device() as device:
        # Create a pipeline
        pipeline = Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setResolution(ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setFps(30)
        stereo = pipeline.createStereoDepth()
        cam_rgb.preview.link(stereo.left)
        stereo.depth.link(pipeline.createXLinkOut().input)

        device.startPipeline(pipeline)

        # Initialize ORB-SLAM2
        orb_slam = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        orb_slam.setFastThreshold(0)

        # Initialize variables for SLAM
        map_points = {}
        last_frame = None
        last_kps, last_des = None, None
        scale = 1.0

        while True:
            # Get the next available frame from the camera
            in_left = device.getOutputQueue(name="left", maxSize=1, blocking=False).get()
            if in_left is not None:
                frame = in_left.getCvFrame()

                # Resize frame for ORB-SLAM2
                frame_small = cv2.resize(frame, None, fx=scale, fy=scale)

                # Detect ORB features and compute descriptors
                kps, des = orb_slam.detectAndCompute(frame_small, None)

                if last_frame is not None:
                    # Match features between consecutive frames
                    matches = bf.match(des, last_des)

                    # Extract matched keypoints and compute essential matrix
                    src_pts = np.float32([kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([last_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    E, mask = cv2.findEssentialMat(dst_pts, src_pts)

                    # Recover pose from essential matrix
                    _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts)

                    # Update map with new points
                    for idx in range(len(kps)):
                        point_3d = stereo.calc_point_cloud(depth_frame, Point2f(kps[idx].pt[0], kps[idx].pt[1]))
                        map_points[kps[idx].pt] = point_3d

                    # Update last frame keypoints and descriptors
                    last_kps, last_des = kps, des

                else:
                    last_kps, last_des = kps, des

                # Store current frame as last frame for next iteration
                last_frame = frame_small

                # Visualize SLAM output (for demonstration purposes)
                for kp in kps:
                    pt = tuple(np.int0(scale * np.array(kp.pt)))
                    cv2.circle(frame, pt, 3, (0, 0, 255), -1)

                # Convert the frame to JPEG format and yield it
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream the video feed from the depth camera
@app.route('/video_feed')
def video_feed():
    return Response(run_slam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to render the web app
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
