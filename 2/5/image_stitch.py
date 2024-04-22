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
    img1_path = '2/5/Image_for_stitch_1.jpg'
    img2_path = '2/5/Image_for_stitch_2.jpg'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    panorama = stitch_2_imgs(img1, img2, "SIFT")

    output_path = '2/5/pano2.jpg'
    cv2.imwrite(output_path, panorama)
    cv2.imshow('Image', panorama)
    cv2.waitKey(0)
