from flask import Flask, render_template, request
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'webApp/static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

click_count = 0
uv1, uv2 = np.zeros((2,)), np.zeros((2,))
K_inv = np.linalg.inv(
    np.array([
        [2335, 0, 1151],
        [0, 2334, 535],
        [0, 0, 1]])
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/estimate_length', methods=['POST'])
def estimate_length():
    global uv1, uv2
    zc = float(request.form['zc'])
    uv1 = np.array([float(coord) for coord in request.form['uv1'].split(',')])
    uv2 = np.array([float(coord) for coord in request.form['uv2'].split(',')])

    xyz1 = K_inv.dot(np.append(uv1, 1)) * zc
    xyz2 = K_inv.dot(np.append(uv2, 1)) * zc
    length = np.linalg.norm((xyz2 - xyz1))

    return str(length)


@app.route('/compute_integral', methods=['POST'])
def compute_integral():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = 'uploaded_image.jpg'  # Change the filename as needed
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            img = cv2.imread(file_path)
            img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            integral_img = compute_integral_image(img_bw)

            # Convert integral image to RGB for display
            integral_img_rgb = cv2.cvtColor(integral_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            integral_img_path = os.path.join(UPLOAD_FOLDER, 'integral_image.jpg')
            cv2.imwrite(integral_img_path, integral_img_rgb)

            return render_template('integral_image.html',
                                   original_img_path=file_path,
                                   integral_img_path=integral_img_path)


def compute_integral_image(img_bw):
    # Initialize integral image
    integral_img = np.zeros_like(img_bw, dtype=np.int32)

    # Compute integral image
    integral_img[0, 0] = img_bw[0, 0]
    for i in range(1, img_bw.shape[0]):
        integral_img[i, 0] = integral_img[i - 1, 0] + img_bw[i, 0]
    for j in range(1, img_bw.shape[1]):
        integral_img[0, j] = integral_img[0, j - 1] + img_bw[0, j]
    for i in range(1, img_bw.shape[0]):
        for j in range(1, img_bw.shape[1]):
            integral_img[i, j] = (integral_img[i - 1, j] + integral_img[i, j - 1] -
                                  integral_img[i - 1, j - 1] + img_bw[i, j])
    return integral_img


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


@app.route('/stitch_images', methods=['POST'])
def stitch_images():
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            return "Missing file(s)"
        file1 = request.files['image1']
        file2 = request.files['image2']
        if file1.filename == '' or file2.filename == '':
            return "No selected file"
        if file1 and file2:
            filename1 = 'uploaded_image1.jpg'  
            filename2 = 'uploaded_image2.jpg' 
            file_path1 = os.path.join(UPLOAD_FOLDER, filename1)
            file_path2 = os.path.join(UPLOAD_FOLDER, filename2)
            file1.save(file_path1)
            file2.save(file_path2)
            img1 = cv2.imread(file_path1)
            img2 = cv2.imread(file_path2)
            stitched_img = stitch_2_imgs(img1, img2, "SIFT")  # Change the feature if needed
            stitched_img_path = os.path.join(UPLOAD_FOLDER, 'stitched_image.jpg')
            cv2.imwrite(stitched_img_path, stitched_img)
            return render_template('stitched_image.html', stitched_img_path=stitched_img_path)


if __name__ == "__main__":
    app.run(debug=True)
