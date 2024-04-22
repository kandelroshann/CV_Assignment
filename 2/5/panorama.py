import depthai
import cv2
import numpy as np

def stitch_2_imgs(img1, img2, feature):
    img1_bw = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_bw = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if feature == "SIFT":
        feat = cv2.SIFT_create() 
    elif feature == "ORB":
        feat = cv2.ORB_create()
    else:
        raise ValueError("Invalid feature. Please choose 'SIFT' or 'ORB'.")

    kp_img1, desc_img1 = feat.detectAndCompute(img1_bw, None) 
    kp_img2, desc_img2 = feat.detectAndCompute(img2_bw, None) 

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_img2, desc_img1, k=2)

    good_points = []
    for m, n in matches: 
        if m.distance < 0.6 * n.distance: 
            good_points.append(m) 

    query_pts = np.float32([kp_img2[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2) 
    train_pts = np.float32([kp_img1[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2) 

    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0) 

    # Warped image
    dst = cv2.warpPerspective(img2, matrix, ((img1.shape[1] + img2.shape[1]), img2.shape[0])) 

    # Paste them together
    dst[0:img1.shape[0], 0:img1.shape[1]] = img1

    # Find non-black pixels indices
    non_black_pixels = np.where(np.any(dst != [0, 0, 0], axis=-1))
    min_x = np.min(non_black_pixels[1])
    max_x = np.max(non_black_pixels[1])

    # Crop the image
    cropped_img = dst[:, min_x:max_x]

    return cropped_img

if __name__ == "__main__":
    # Initialize DepthAI pipeline
    pipeline = depthai.Pipeline()
    cam = pipeline.create(depthai.node.ColorCamera)
    cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    xout = pipeline.create(depthai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)
    
    # Start the pipeline
    with depthai.Device(pipeline) as device:
        num_images_captured = 0
        captured_images = []

        # Main loop
        while num_images_captured < 3:
            data = device.getOutputQueue("video").get()
            img = data.getCvFrame()
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)

            if key != -1:
                img_path = f"Image_for_stitch_{num_images_captured + 1}.jpg"
                cv2.imwrite(img_path, img)
                captured_images.append(cv2.imread(img_path))
                num_images_captured += 1
                print(f"Image {num_images_captured} captured.")

        cv2.destroyAllWindows()

        # Stitch captured images
        panorama = captured_images[0]
        for i in range(1, len(captured_images)):
            panorama = stitch_2_imgs(panorama, captured_images[i], "SIFT")

        output_path = 'panorama.jpg'
        cv2.imwrite(output_path, panorama)
        cv2.imshow('Panorama', panorama)
        cv2.waitKey(0)


        # Save captured images
        for i, img in enumerate(captured_images):
            cv2.imwrite(f"captured_image_{i+1}.jpg", img)
