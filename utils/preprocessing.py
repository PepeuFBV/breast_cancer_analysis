import matplotlib as plt
import cv2
import numpy as np
import itertools
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
    # Only pass tileGridSize to createCLAHE
    clahe = cv2.createCLAHE(tileGridSize=tile_grid_size)
    result = image.copy()
    for _ in range(iterations):
        result = clahe.apply(result)
    if show:
        cv2.imshow('CLAHE', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result


def add_all_2_method_combinations(preprocessing_methods):
    method_names = list(preprocessing_methods.keys())
    for m1, m2 in itertools.permutations(method_names, 2):
        func1 = preprocessing_methods[m1]['func']
        params1 = preprocessing_methods[m1]['params']
        func2 = preprocessing_methods[m2]['func']
        params2 = preprocessing_methods[m2]['params']

        def combined_func(image, params1, params2, func1=func1, func2=func2):
            return func2(func1(image, **params1), **params2)

        combo_name = f"{m1}__{m2}"

        # build param dictionaries for each method 
        param_dicts1 = [dict(zip(params1.keys(), values)) for values in itertools.product(*params1.values())]
        param_dicts2 = [dict(zip(params2.keys(), values)) for values in itertools.product(*params2.values())]

        preprocessing_methods[combo_name] = {
            'func': combined_func,
            'params': {
                f"{m1}_params": param_dicts1,
                f"{m2}_params": param_dicts2
            }
        }
        

preprocessing_methods = { # 203 total combinations
    'denoise': { # 25 combinations
        'func': denoise_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (9, 9)],#, (11, 11), (15, 15)],
            'sigma': [0, 1, 3, 5],#, 7]
        }
    },
    'binarize': { # 48 combinations
        'func': binarize_image,
        'params': {
            'threshold': [70, 127, 200, 255], 
            'max_value': [70, 127, 200, 255],
            'method': ['fixed', 'adaptive_mean', 'adaptive_gaussian']
        }
    },
    'lowpass': { # 5 combinations
        'func': lowpass_filter_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (9, 9)],#, (11, 11), (15, 15)]
        }
    },
    'erode': { # 25 combinations
        'func': erode_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (9, 9)],#, (11, 11)],#, (15, 15)],
            'iterations': [1, 2, 3]#, 5, 7]
        }
    },
    'dilate': { # 25 combinations
        'func': dilate_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (9, 9)],#, (11, 11)],#, (15, 15)],
            'iterations': [1, 2, 3]#, 5, 7]
        }
    },
    'open': { # 25 combinations
        'func': open_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (9, 9)],#, (11, 11)],#, (15, 15)],
            'iterations': [1, 2, 3]#, 5, 7]
        }
    },
    'close': { # 25 combinations
        'func': close_image,
        'params': {
            'kernel_size': [(3, 3), (5, 5), (9, 9)],#, (11, 11)],#, (15, 15)],
            'iterations': [1, 2, 3]#, 5, 7]
        }
    },
    'clahe': { # 25 combinations
        'func': clahe_image,
        'params': {
            'tile_grid_size': [(3, 3), (5, 5), (9, 9)],#, (11, 11)],#, (15, 15)],
            'iterations': [1, 2, 3]#, 5, 7]
        }
    }
}

add_all_2_method_combinations(preprocessing_methods) # adding all combinations of two methods (will create a ton of new methods)
