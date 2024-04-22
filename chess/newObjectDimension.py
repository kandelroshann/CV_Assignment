import cv2
import numpy as np

# Read the image
image = cv2.imread('/Users/roshankandel/Downloads/CVAssignment/chess/Image_1.jpg')

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Select two points on the object
print("Select two points on the object (click twice)")
points = np.zeros((2, 2), dtype=int)
def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points[params[0]] = [x, y]
        params[0] += 1
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Image', image)

cv2.setMouseCallback('Image', mouse_callback, [0])

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Measure the distance between the selected points in pixels
pixel_distance = np.linalg.norm(points[1] - points[0])

# Specify the distance between the camera and the object in meters
camera_distance = 0.45  # You can adjust this according to your setup

# Calculate the real-world dimension of the object using perspective projection equation
# Assuming the object is at a right angle to the camera
# If the object is not at a right angle, more complex calculations are needed
# You can also adjust this equation according to your camera's specifications
object_dimension_meters = 2 * camera_distance * np.tan(np.arctan(pixel_distance / 2))

# Display the calculated real-world dimension
print('The real-world dimension of the object is approximately', object_dimension_meters, 'meters')
