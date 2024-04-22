import cv2
from flask import Flask, Response, jsonify
from depthai import Device, Pipeline, CameraBoardSocket, ColorCameraProperties

app = Flask(__name__)

# Function to detect and track objects in a frame
def detect_and_track(frame=None):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 30, 150)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tracked_objects = {}
    
    # Loop through contours to find bounding boxes
    for contour in contours:
        # Compute the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Consider only contours with a minimum area
        if cv2.contourArea(contour) > 500:
            # Calculate the dimensions of the bounding box
            width = w
            height = h
            
            # Append the dimensions to the tracked_objects dictionary
            tracked_objects[len(tracked_objects)] = {
                'bbox': (x, y, x + w, y + h),
                'dimensions': (width, height)
            }
    
    return tracked_objects

# Function to generate video frames from the OAK-D camera
def generate_frames():
    with Device() as device:
        pipeline = Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(CameraBoardSocket.RGB)
        cam_rgb.setResolution(ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setFps(30)
        cam_rgb_xout = pipeline.createXLinkOut()
        cam_rgb_xout.setStreamName("rgb")
        cam_rgb.preview.link(cam_rgb_xout.input)

        device.startPipeline(pipeline)

        while True:
            # Get the next available frame from the OAK-D camera
            in_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False).get()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                
                # Detect and track objects in the frame
                tracked_objects = detect_and_track(frame)
                
                # Draw bounding boxes around tracked objects
                for obj_id, obj_data in tracked_objects.items():
                    bbox = obj_data['bbox']
                    dimensions = obj_data['dimensions']
                    (x, y, x2, y2) = bbox
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {obj_id}, Dimensions: {dimensions}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convert the frame to JPEG format and yield it
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream the video feed from the OAK-D camera
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to provide object information (dimensions) as JSON
@app.route('/object_info')
def object_info():
    tracked_objects = detect_and_track()  # Call detect_and_track function here
    object_info_list = []
    
    for obj_id, obj_data in tracked_objects.items():
        bbox = obj_data['bbox']
        dimensions = obj_data['dimensions']
        
        object_info_list.append({
            'id': obj_id,
            'bbox': bbox,
            'dimensions': dimensions
        })
    
    return jsonify(object_info_list)

# Route to render the web app
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OAK-D Camera Feed</title>
    </head>
    <body>
        <h1>OAK-D Camera Feed</h1>
        <img src="/video_feed" width="640" height="480">
        <div id="object_info"></div>
        <script>
            setInterval(function() {
                fetch('/object_info')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('object_info').innerText = JSON.stringify(data);
                    });
            }, 1000);  // Update every second
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
