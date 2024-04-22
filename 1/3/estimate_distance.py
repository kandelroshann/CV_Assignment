import numpy as np
import matplotlib.pyplot as plt

click_count = 0
uv1, uv2 = np.zeros((3, 1)), np.zeros((3, 1))
K_inv = np.linalg.inv(
    np.array([
        [2553, 0, 1152],
        [0, 2553, 530],
        [0, 0, 1]])
)


def onclick(event):
    global uv1, uv2, click_count
    homogeneous_pixel_coords = np.array([[event.xdata], [event.ydata], [1]])
    if click_count == 0:
        uv1 = homogeneous_pixel_coords
    else:
        uv2 = homogeneous_pixel_coords
        plt.close()
    click_count += 1


def main():
    global uv1, uv2
    image = plt.imread('chess/Image_1.jpg')  
    zc = input('Distance from camera (in cm): ')
    zc = float(zc)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title('Click to print coordinates')
    ax.axis('off')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    plt.axis('off')
    plt.imshow(image)
    plt.scatter(uv1[0,0], uv1[1,0], color='yellow', s=100, zorder=2)
    plt.scatter(uv2[0,0], uv2[1,0], color='yellow', s=100, zorder=2)
    xyz1 = K_inv.dot(uv1) * zc
    xyz2 = K_inv.dot(uv2) * zc
    length = np.linalg.norm((xyz2 - xyz1)) 
    plt.title(f'Estimated length: {length:.2f} cm')
    plt.show()


if __name__ == '__main__':
    main()
