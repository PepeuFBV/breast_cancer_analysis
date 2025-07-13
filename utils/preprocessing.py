import matplotlib as plt
import cv2
import numpy as np
import sys
sys.path.append('../utils')


def denoise_image(image, kernel_size=(5, 5), sigma=0, show=False):
    """
    Denoise an image using Gaussian blur.
    Parameters:
        image (numpy.ndarray): Input image to be denoised.
        kernel_size (tuple): Size of the Gaussian kernel. Default is (5, 5).
        sigma (float): Standard deviation for Gaussian kernel. Default is 0.
        show (bool): If True, display the denoised image. Default is False.
    Returns:
        numpy.ndarray: Denoised image.
    """
    if show:
        denoised = cv2.GaussianBlur(image, kernel_size, sigma)
        plt.imshow(denoised, cmap='gray')
        plt.title('Denoised Image')
        plt.axis('off')
        plt.show()
        return denoised
    else: 
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    
def binarize_image(image, threshold=127, max_value=255, method='fixed', show=False):
    """
    Binarize an image using different methods.
    Parameters:
        image (numpy.ndarray): Input image to be binarized.
        threshold (int): Threshold value for binarization. Default is 127.
        max_value (int): Maximum value to use with binary thresholding. Default is 255
        method (str): Method for binarization. Options are 'fixed', 'adaptive_mean', 'adaptive_gaussian'. Default is 'fixed'.
        show (bool): If True, display the binarized image. Default is False.
    Returns:
        numpy.ndarray: Binarized image.
    """
    if method == 'fixed':
        _, binary = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    elif method == 'adaptive_mean':
        binary = cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'adaptive_gaussian':
        binary = cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        raise ValueError("Unknown method: choose 'fixed', 'adaptive_mean', or 'adaptive_gaussian'")
    if show:
        plt.imshow(binary, cmap='gray')
        plt.title('Binarized Image')
        plt.axis('off')
        plt.show()
    return binary


def lowpass_filter_image(image, kernel_size=(5, 5), show=False):
    """
    Apply a low-pass filter (Gaussian blur) to an image.
    Parameters:
        image (numpy.ndarray): Input image to be filtered.
        kernel_size (tuple): Size of the Gaussian kernel. Default is (5, 5).
        show (bool): If True, display the filtered image. Default is False.
    Returns:
        numpy.ndarray: Low-pass filtered image.
    """
    blurred = cv2.blur(image, kernel_size)
    if show:
        plt.imshow(blurred, cmap='gray')
        plt.title('Low-pass Filtered Image')
        plt.axis('off')
        plt.show()
    return blurred


def erode_image(image, kernel_size=(3, 3), iterations=1, show=False):
    """
    Erode an image using a morphological operation.
    Parameters:
        image (numpy.ndarray): Input image to be eroded.
        kernel_size (tuple): Size of the structuring element. Default is (3, 3).
        iterations (int): Number of times erosion is applied. Default is 1.
        show (bool): If True, display the eroded image. Default is False.
    Returns:
        numpy.ndarray: Eroded image.
    """
    kernel = np.ones(kernel_size, np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    if show:
        plt.imshow(eroded, cmap='gray')
        plt.title('Eroded Image')
        plt.axis('off')
        plt.show()
    return eroded


def dilate_image(image, kernel_size=(3, 3), iterations=1, show=False):
    """
    Dilate an image using a morphological operation.
    Parameters:
        image (numpy.ndarray): Input image to be dilated.
        kernel_size (tuple): Size of the structuring element. Default is (3, 3).
        iterations (int): Number of times dilation is applied. Default is 1.
        show (bool): If True, display the dilated image. Default is False.
    Returns:
        numpy.ndarray: Dilated image.
    """
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    if show:
        plt.imshow(dilated, cmap='gray')
        plt.title('Dilated Image')
        plt.axis('off')
        plt.show()
    return dilated


def open_image(image, kernel_size=(3, 3), iterations=1, show=False):
    """
    Open an image using a morphological operation (erosion followed by dilation).
    Parameters:
        image (numpy.ndarray): Input image to be opened.
        kernel_size (tuple): Size of the structuring element. Default is (3, 3).
        iterations (int): Number of times the operation is applied. Default is 1.
        show (bool): If True, display the opened image. Default is False.
    Returns:
        numpy.ndarray: Opened image.
    """
    kernel = np.ones(kernel_size, np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if show:
        plt.imshow(opened, cmap='gray')
        plt.title('Opened Image')
        plt.axis('off')
        plt.show()
    return opened


def close_image(image, kernel_size=(3, 3), iterations=1, show=False):
    """
    Close an image using a morphological operation (dilation followed by erosion).
    Parameters:
        image (numpy.ndarray): Input image to be closed.
        kernel_size (tuple): Size of the structuring element. Default is (3, 3).
        iterations (int): Number of times the operation is applied. Default is 1.
        show (bool): If True, display the closed image. Default is False.
    Returns:
        numpy.ndarray: Closed image.
    """
    kernel = np.ones(kernel_size, np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    if show:
        plt.imshow(closed, cmap='gray')
        plt.title('Closed Image')
        plt.axis('off')
        plt.show()
    return closed


def clahe_image(image, tile_grid_size=(3, 3), iterations=1, show=False):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    Parameters:
        image (numpy.ndarray): Input image to be enhanced.
        tile_grid_size (tuple): Size of the grid for histogram equalization. Default is (3, 3).
        iterations (int): Number of times the operation is applied. Default is 1.
        show (bool): If True, display the enhanced image. Default is False.
    Returns:
        numpy.ndarray: Enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image)
    if show:
        plt.imshow(clahe_image, cmap='gray')
        plt.title('CLAHE Image')
        plt.axis('off')
        plt.show()
    return clahe_image


preprocessing_methods = {
    'denoise': {
        'func': denoise_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (7, 7)],
            'sigma': [0, 1, 2]
        }
    },
    'binarize': {
        'func': binarize_image,
        'params': {
            'threshold': [100, 127, 150],
            'max_value': [255],
            'method': ['fixed', 'adaptive_mean', 'adaptive_gaussian']
        }
    },
    'lowpass_filter': {
        'func': lowpass_filter_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (7, 7)]
        }
    },
    'erode': {
        'func': erode_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (7, 7)],
            'iterations': [1, 2, 3, 5]
        }
    },
    'dilate': {
        'func': dilate_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (7, 7)],
            'iterations': [1, 2, 3, 5]
        }
    },
    'open': {
        'func': open_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (7, 7)],
            'iterations': [1, 2, 3, 5]
        }
    },
    'close': {
        'func': close_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (7, 7)],
            'iterations': [1, 2, 3, 5]
        }
    },
    'clahe': {
        'func': clahe_image,
        'params': {
            'tile_grid_size': [(3, 3), (5, 5), (7, 7)],
            'iterations': [1, 2, 3, 5]
        }
    }
}
