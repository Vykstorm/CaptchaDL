

import numpy as np
import pandas as pd
import cv2 as cv
from math import floor

from input import InputFlow
from dataset import CaptchaDataset
from char import find_chars


'''
This script will generate a dataset which consists of individual character
images (extracted from captcha dataset samples) and their corresponding label
'''

# Number of samples to generate for the dataset
NUM_SAMPLES = 50000

# Image dimensions for each sample: height x width
IMAGE_SIZE = (40, 40)


class CharImageGenerator(InputFlow):
    '''
    This class exposes the iterator interface and yields the captcha characters and its
    labels
    '''
    def __init__(self):
        dataset = CaptchaDataset()
        num_samples, text_size = dataset.num_samples, dataset.text_size

        extra_samples = num_samples
        while (num_samples + extra_samples) * text_size < NUM_SAMPLES:
            extra_samples += num_samples

        super().__init__(
            dataset.X, dataset.y, batch_size=1, shuffle=True,
            generate_samples=extra_samples
        )


    def __iter__(self):
        text_size = CaptchaDataset().text_size

        it = super().__iter__()
        while True:
            X_batch, y_batch = next(it)

            chars = find_chars(X_batch[0, :, :, 0], char_size=IMAGE_SIZE, num_chars=text_size)
            for k in range(0, text_size):
                yield chars[k], y_batch[0, k, :]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    # Generate char samples
    generator = iter(CharImageGenerator())

    X = np.zeros([NUM_SAMPLES] + list(IMAGE_SIZE) + [1]).astype(np.float32)
    y = np.zeros([NUM_SAMPLES, CaptchaDataset().num_char_classes]).astype(np.uint8)


    epochs_per_tick = 10
    print()
    for k in range(0, NUM_SAMPLES):
        if k % epochs_per_tick == 0:
            print('{}/{}, {}%'.format(k, NUM_SAMPLES, floor(k / NUM_SAMPLES * 100)).rjust(18), end='\r')
        X[k, :, :, 0], y[k, :] = next(generator)
    print('', end='\r')
    print()
    print('Done')

    # Print info
    df = pd.DataFrame.from_dict({
        'attr': ['Number of samples', 'Image dimensions', 'Number of char classes'],
        'values': [X.shape[0], X.shape[1:], y.shape[1]]
    })
    df.set_index('attr', inplace=True)
    print(df)

    # Save dataset
    np.savez_compressed('.chars.npz', X=X, y=y)
