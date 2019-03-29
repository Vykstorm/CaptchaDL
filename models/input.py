'''
This script provides helper methods & classes to provide the features & labels
from the dataset for the deep learning models train/evaluate methods
'''

from itertools import product, islice, chain
from functools import lru_cache

import keras
from keras.layers import Input, Flatten, Dense
from keras.models import Model

import numpy as np
from scipy.optimize import minimize
import sklearn.utils



class ImageGenerator(keras.preprocessing.image.ImageDataGenerator):
    '''
    This class is used to amplify the number of images without the need of
    additional labels by applying affine transformations like shearing or
    shifting
    Its a wrapper around keras ImageDataGenerator class
    '''
    def __init__(self):
        super().__init__(
            width_shift_range=0.1,
            shear_range=0.15,
            rotation_range=7)


class InputFlow:
    '''
    This class acts like a pipeline between the dataset and the deep learning models
    It takes a set of samples and generates batches that later can be used on each
    model train epoch step
    '''
    def __init__(self, X, y, batch_size=32, shuffle=True, random_state=None, generate_samples=0):
        '''
        Initializes this instance
        :param X: A 4D array of size n.samples x height x width x n.channels with the image features
        :param y: The categoric labels of the samples. (an
        :param shuffle: Set it to true to shuffle the samples before dispatching the batches
        :param random_state: Random state to shuffle the samples
        :param generate_samples: Its an optional integer. It sets the number of additional samples
        to generate. This will be rounded up if the number of samples plus this amount is not a multiple
        of the batch size
        '''
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.generate_samples = generate_samples
        self.image_generator = ImageGenerator()


    @property
    @lru_cache(maxsize=1)
    def repetitions(self):
        y = self.y
        batch_size, generate_samples = self.batch_size, self.generate_samples


        if generate_samples == 0:
            return np.ones([y.shape[0]]).astype(np.uint32)

        def normalize(x, bounds=(0, 1)):
            return ((x - x.min()) / (x.max() - x.min())) * bounds[1] + bounds[0]

        def clamp(x, bounds=(0, 1)):
            return np.maximum(np.minimum(x, bounds[1]), bounds[0])

        # Set numpy random state
        np.random.seed(self.random_state)

        # Get char frequencies
        char_f = y.sum(axis=1).sum(axis=0)
        char_f_nz = char_f[char_f > 0]
        char_f = clamp(char_f, (char_f_nz.min(), char_f_nz.max()))
        char_f_score = 1 - np.power(normalize(char_f), 1)

        # Calculate samples ranking
        rankings = normalize(np.multiply(y, char_f_score.reshape(1, 1, -1)).sum(axis=2).sum(axis=1))


        # Now calculate the number of repetitions for each sample
        n = y.shape[0]
        n2 = n + generate_samples

        f = lambda x: x * rankings.sum() - n2
        loss = lambda x: f(x) ** 2
        loss_derivative = lambda x: 2 * f(x) * rankings.sum()

        result = minimize(loss,
              x0=np.ones([1]),
              jac=loss_derivative)

        repetitions = np.ceil(result.x * rankings).astype(np.uint32)
        repetitions[0:(batch_size - int(repetitions.sum()) % batch_size)] += 1
        return repetitions


    @property
    @lru_cache(maxsize=1)
    def indices(self):
        y, repetitions = self.y, self.repetitions

        indices = []
        for i in range(0, y.shape[0]):
            indices.extend([i] * repetitions[i])
        indices = np.array(indices, dtype=np.uint32)
        return indices

    def __len__(self):
        '''
        Returns the amount to be specified in the parameter 'steps_per_epoch' when
        using fit_generator()
        '''
        return int(self.repetitions.sum().item()) // self.batch_size

    def __iter__(self):
        '''
        Returns an iterator which generates tuples of pairs X_batch, y_batch
        where X_batch are the images of the samples on the batch and y_batch are
        its corresponding categorical labels
        '''
        X, y = self.X, self.y
        batch_size = self.batch_size
        image_generator = self.image_generator
        indices = self.indices

        # Dispatch the samples packed on batches
        while True:
            # Shuffle the samples at each epoch
            if self.shuffle:
                indices = sklearn.utils.shuffle(indices, random_state=self.random_state)

            for i in range(0, y.shape[0], batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                yield from islice(image_generator.flow(X_batch, y_batch, batch_size=batch_size, shuffle=False), 1)



if __name__ == '__main__':
    from dataset import CaptchaDataset
    import pandas as pd

    dataset = CaptchaDataset()
    it = iter(InputFlow(dataset.X, dataset.y, generate_samples=4000 ))
    X_batch, y_batch = next(it)

    print('Batch shapes: X {}, y {}'.format(X_batch.shape, y_batch.shape))
    texts = dataset.labels_to_text(y_batch.argmax(axis=2))
    print('Captions: ', texts)

    H = chain([('char frequencies', 'value')],
            zip(dataset.alphabet,
                [[x] for x in np.histogram(y_batch.argmax(axis=2).flatten(), bins=y_batch.shape[2])[0]]
            )
        )
    df = pd.DataFrame.from_dict(dict(list(H)))
    df.set_index('char frequencies', drop=True, inplace=True)
    print(df)
