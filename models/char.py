


from contours import find_contours
import numpy as np
import pandas as pd
import cv2 as cv
from math import floor, ceil
from functools import partial
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from itertools import product, repeat, permutations, combinations_with_replacement, chain



def find_char_separators(img, num_chars=2):
    '''
    Find image vertical lines to delimit the characters inside it

    Returns a 1D array with integer values in the range [0, img.shape[1]).
    The number of returned delimiters are num_chars-1

    Each delimiter indicates the char frame offset horizontally
    '''
    f = (img > 0).mean(axis=0)
    a, b = f.min(), f.max()
    f = (f - a) / (b - a)
    n, k = len(f), num_chars
    x = np.arange(0, n)

    # Initial guess
    s0 = np.linspace(0, n, k+1)[1:-1]

    # Value boundaries for each delimiter.
    char_min_size = 15
    delimiter_margin = 8
    bounds = np.transpose(np.stack([
                            np.maximum(s0 - delimiter_margin, 0),
                            np.minimum(s0 + delimiter_margin, n-1)],
                        axis=0))

    # Now we move the delimiters to divide the chars better

    y_spl = UnivariateSpline(x, f, s=0,k=4)
    y_spl_df = y_spl.derivative(n=1)

    F = lambda s: np.sum(y_spl(s))
    dF = lambda s: y_spl_df(s) / (k - 1)
    jac = lambda s, *args: dF(s)

    result = minimize(F, s0, jac=jac, method='SLSQP', bounds=bounds)
    s = np.round(result.x)

    separators = s.astype(np.uint16)
    return separators



def process_image(img, dsize):
    '''
    This method takes an image and process it (its optimized for images that contains
    characters inside from the captcha dataset):
    - It tries to homogenize the background color by painting gray scaled colors that
    not belongs to the foreground with white

    - Then its resized to fit the size specified (if the image is smaller, borders with the
    same color as the background will be added on each direction)
    '''

    # Process the image
    inverted = 255 - img
    ret, thresholded = cv.threshold(inverted, 70, 255, cv.THRESH_BINARY)
    img = 255 - np.multiply((thresholded > 0), inverted)

    # Resize the image
    dh, dw = dsize
    h, w = img.shape

    if w < dw:
        left = floor((dw - w) / 2)
        right = dw - w - left
        img = cv.copyMakeBorder(img, 0, 0, left, right, cv.BORDER_CONSTANT, value=(255, 255, 255))
    elif w > dw:
        left = floor((w - dw) / 2)
        img = img[:, left:left+dw]

    if h < dh:
        top = floor((dh - h) / 2)
        bottom = dh - h - top
        img = cv.copyMakeBorder(img, top, bottom, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
    elif h > dh:
        top = floor((h - dh) / 2)
        img = img[top:top+dh, :]

    return img



def split_array(a, separators, axis=1):
    '''
    This method splits an array in multiple subarray over the given axis
    using the separators
    '''
    seperators = sorted(separators)
    n_sep = len(separators)

    if n_sep == 1:
        sep = separators[0]
        a = a.swapaxes(0, axis)
        return [a[0:sep].swapaxes(0, axis), a[sep:].swapaxes(0, axis)]

    head, body = split_array(a, [separators[0]], axis)
    splits = split_array(body, np.array(separators[1:]) - separators[0], axis)
    return [head] + splits



def find_chars(img, char_size, num_chars=5):
    '''
    This function takes a gray scaled image and detects text characters
    (Is optimized for the captcha dataset)
    :param img: Must be a gray scaled image (2D array with float32 values in the range [0, 1])
    :param num_chars: Number of characters to be extracted from the image
    :return Returns a 3D array of size num_chars x n x m
    Where n and m indicates the shape of the output images (can be specified using char_size)
    '''
    # Extract image contours
    contours = find_contours(img)

    # Remove contours which we predict that dont have any char inside
    contours = [contour for contour in contours if contour.num_chars > 0]
    assert len(contours) > 0

    k = len(contours)

    # Sort frames by its horizontal position (from left to right)
    contours.sort(key=lambda contour: contour.bbox.left)

    # Now we create a 2D matrix where the element at index i,j
    # will be the probability of the frame i to contain j characters inside
    P = np.array([contour.num_chars_proba for contour in contours])

    # If n0, n1, ..., nk are the number of predicted characters inside each frame, we find the best configuration so that n0 + n1 + ... + nk = num_chars
    # and ensure that P[0, n[0]] * P[1, n[1]] * ... * P[k, n[k]] is maximized

    # All valid configurations (n0, n1, ..., nk) such that n0 + n1 + ... + nk = num_chars
    configs = filter(lambda x: np.sum(x) == num_chars, combinations_with_replacement(range(0, num_chars+1), k))
    configs = list(chain.from_iterable(map(lambda config: permutations(config, k), configs)))
    assert len(configs) > 0

    configs = np.array(configs, dtype=np.uint8)
    nc = configs.shape[0]

    # Calculate a score function for each configuration
    scores = np.zeros([nc]).astype(np.float32)

    for i in range(0, nc):
        scores[i] = np.prod(P[np.arange(0, k), np.array(configs[i])])

    # Get the best configuration
    best_config = configs[np.argmax(scores)]

    # Split the contours into frames
    img = (img * 255).astype(np.uint8)
    frames = []

    for k in range(0, k):
        if best_config[k] == 0:
            continue

        elif best_config[k] == 1:
            # Contour boundaries only holds 1 char.
            frame = contours[k].extract_bbox_pixels(img)
            frames.append(frame)
        else:
            # Contour holds more than 1 char
            # Split it into multiple frames
            separators = find_char_separators(contours[k].bbox_mask, best_config[k])
            splits = split_array(contours[k].extract_bbox_pixels(img), separators, axis=1)
            frames.extend(splits)

    processed_frames = map(partial(process_image, dsize=char_size), frames)
    return [frame.astype(np.float32) / 255 for frame in processed_frames]


if __name__ == '__main__':
    # Unitary test

    from input import InputFlow
    from dataset import CaptchaDataset
    import matplotlib.pyplot as plt
    from utils import waitKey

    def wait():
        if waitKey() == 'q':
            raise KeyboardInterrupt()

    try:
        while True:
            # Get 1 sample from the captcha dataset
            dataset = CaptchaDataset()
            input = iter(InputFlow(dataset.X, dataset.y, batch_size=1))
            X_batch, y_batch = next(input)

            # Find characters in the image
            frames = find_chars(X_batch[0, :, :, 0], char_size=(40, 40), num_chars=dataset.text_size)


            # Show all the extracted characters

            n = len(frames)
            fig, ax = plt.subplots(1, n, figsize=(10, 2))

            for i in range(0, n):
                plt.sca(ax[i])
                plt.imshow(frames[i], cmap='gray')
                plt.title('{}th'.format(i+1))
                plt.xticks([])
                plt.yticks([])

            plt.tight_layout()
            plt.show()

    except KeyboardInterrupt:
        pass
