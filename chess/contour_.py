import cv2

def find_real_dimensions(image, focal_length_x, focal_length_y, distance_cm):
    """
    Calculate real-world dimensions of an object using perspective projection equations.

    Parameters:
    - image: numpy array representing the image
    - focal_length_x: focal length of the camera in the x-axis in pixels
    - focal_length_y: focal length of the camera in the y-axis in pixels
    - distance_cm: distance from the camera to the object in centimeters

    Returns:
    - real_dimensions: tuple (width_cm, height_cm) of the object in the real world in centimeters
    """

    # Display the image and allow the user to select the object
    clone = image.copy()
    cv2.imshow("Select Object", clone)
    roi = cv2.selectROI("Select Object", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Extracting object dimensions
    object_width, object_height = roi[2], roi[3]

    # Calculating real dimensions using perspective projection equations
    real_width_cm = (object_width * distance_cm) / focal_length_x
    real_height_cm = (object_height * distance_cm) / focal_length_y

    return real_width_cm, real_height_cm

# Load the image
image = cv2.imread("/Users/roshankandel/Downloads/CVAssignment/chess/Image_17.jpg")

# Example values
focal_length_x = 5245  # focal length of the camera in the x-axis in pixels
focal_length_y = 5918  # focal length of the camera in the y-axis in pixels
distance_cm = 30  # distance from the camera to the object in centimeters

# Find real dimensions of the object in the image
real_dimensions_cm = find_real_dimensions(image, focal_length_x, focal_length_y, distance_cm)
print("Real dimensions of the object (width_cm, height_cm):", real_dimensions_cm)
