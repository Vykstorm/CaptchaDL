
'''
Some helper methods are defined in this script
'''


import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
import cv2 as cv



def waitKey(t=-1):
    '''
    This method is the same as OpenCV waitKey() method, but it returns
    a char object instead of an integer
    '''
    return chr(cv.waitKey(t) & 255)


def display_batch(X_batch, y_batch):
    '''
    This function displays on a window multiple images with their corresponding
    labels
    :param X_batch: Must be the batch image samples
    :param y_batch: Must be the batch categorical labels
    Returns the figure and the axis of the graph
    '''

    from dataset import CaptchaDataset
    dataset = CaptchaDataset()
    texts = dataset.labels_to_text(y_batch.argmax(axis=2))

    n = X_batch.shape[0]

    # Number of column subplots per row
    cols = ceil(sqrt(n))

    # Number of rows
    rows = n // cols
    if n % cols > 0:
        rows += 1

    # Create rows x cols subplots
    fig, ax = plt.subplots(rows, cols, figsize=(8, 8))

    for i in range(0, rows):
        for j in range(0, cols):
            if i < rows - 1 or n % cols == 0 or j < n % cols:
                index = i * cols + j
                plt.sca(ax[i, j])
                plt.imshow(X_batch[index, :, :, 0] * 255, cmap='gray')
                plt.xticks([])
                plt.yticks([])

                title = b''.join(texts[index]).decode()
                plt.title(title)
            else:
                ax[i, j].set_visible(False)


    plt.tight_layout()
    plt.show()

    return fig, ax
