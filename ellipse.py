import cv2
import matplotlib.pyplot as plt

def refine_binary_image(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def find_largest_contour(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None

def fit_ellipse(contour):
    return cv2.fitEllipse(contour) if contour is not None else None

def plot_ellipse(img, ellipse):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(output_img, ellipse, (0, 0, 255), 2)
    return output_img

def plot_ellipses(img, ellipse):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    output_img = plot_ellipse(img, ellipse)
    ax[1].imshow(output_img)
    ax[1].set_title('Ellipse')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

img = cv2.imread('save/000_HC_segmented.png', cv2.IMREAD_GRAYSCALE)

binary_img = refine_binary_image(img)
largest_contour = find_largest_contour(binary_img)
ellipse = fit_ellipse(largest_contour)

if ellipse:
    center, axes, angle = ellipse
    print("Ellipse Parameters:")
    print("Center:", center)
    print("Axes:", axes)
    print("Angle:", angle)

    plot_ellipses(img, ellipse)
else:
    print("No valid ellipse found.")
