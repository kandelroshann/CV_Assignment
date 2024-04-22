import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def normalize(image):
    # Normalize the image
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return image_normalized.astype(np.uint8)

def main():
    # Read the image
    img = cv2.imread('Ball1.jpg')
    if img is None:
        print("Error: Could not read the image.")
        return

    # Compute integral image
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    integral_img = compute_integral_image(img_bw)

    # Print bottom-right 5x5 patch of the integral image
    print('Integral bottom right 5x5 patch:\n', integral_img[-5:, -5:])

    # Normalize the integral image
    integral_normalized = normalize(integral_img)

    # Print bottom-right 5x5 patch of the normalized integral image
    print('Normalized bottom right 5x5 patch:\n', integral_normalized[-5:, -5:])

    # Display both original RGB image and integral image side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display RGB image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Display normalized integral image
    axes[1].imshow(integral_normalized, cmap='gray')
    axes[1].set_title('Normalized Integral Image')
    axes[1].axis('off')
    
    plt.show()

    # Save normalized integral image to a text file
    np.savetxt('normalized_integral_matrix.txt', integral_normalized, fmt='%d')

if __name__ == "__main__":
    main()
