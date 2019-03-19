

'''
This module is intented to provide helper methods & classes to load the captha text
datasets

You can do the next 3 lines of code to get the captcha images & its labels:
from dataset import CaptchaDataset
dataset = CaptchaDataset()
X, y = dataset.X, dataset.y
'''

from os import listdir
from os.path import isdir, isfile, join

from re import match
from itertools import product, count, chain

from functools import lru_cache

import numpy as np
import skimage
from keras.utils import to_categorical
import sklearn
import sklearn.model_selection

from utils.singleton import singleton
from utils.dictnamespace import DictNamespace

from configobj import ConfigObj as Config
from validate import Validator as ConfigValidator, is_int_list, is_string
from validate import ValidateError
from config import global_config

@singleton
class CaptchaDataset:
    def __init__(self):
        pass

    @property
    @lru_cache(maxsize=1)
    def config(self):
        '''
        This property returns a dictionary with all the configuration variables:
        DATASET_DIR, CAPTCHA_TEXT_SIZE, IMAGE_DIMS
        For more info, read the dataset.conf file.
        '''
        # Helper methods to check config variables
        def check_image_dims(value):
            dims = is_int_list(value, min=3, max=3)
            for dim in dims:
                if dim <= 0:
                    raise ValidateError('Wrong image dimensions specified: {}'.format(dims))
            if dims[2] not in range(1, 4):
                raise ValidateError('Wrong number of image channels ({})'.format(dims[2]))
            return dims

        def check_dir_exists(value):
            value = is_string(value)
            if not isdir(value):
                raise ValidateError('Directory {} doesnt exist'.format(value))
            return value

        # Load and validate the configuration
        config = Config(join('config', 'dataset.conf'),
                        configspec=join('config', 'dataset.spec.conf'),
                        stringify=True)
        result = config.validate(ConfigValidator({
            'image_dims': check_image_dims,
            'dir': check_dir_exists
        }), preserve_errors=True)

        # Raise an error if the configuration is not valid
        if result is not True:
            raise Exception('Invalid dataset configuration file: {}'.format(result))

        # Return the configuration vars
        config = DictNamespace(config, recursive=True)
        return config




    @property
    @lru_cache(maxsize=1)
    def data(self):
        '''
        This method returns a dictionary with all the data in the dataset
        It holds the next entries: X, y, y_labels, alphabet
        '''

        def load_data():
            config = self.config
            text_size = config.CAPTCHA_TEXT_SIZE
            data_dir = config.DATASET_DIR
            image_dims = config.IMAGE_DIMS

            # Get a list of images to be loaded
            images = listdir(data_dir)

            # Filter the images (only load those with a valid basename)
            images = list(filter(lambda image: match('^[a-z0-9]+\..+$', image), images))

            # Extract captcha texts from the filenames
            texts = [match('^([a-z0-9]+)\..+$', image).group(1) for image in images]

            # Get the all the characters used in the dataset
            alphabet = list(frozenset(chain.from_iterable(texts)))
            alphabet.sort()

            # Generate image labels
            y_labels = np.zeros([len(texts), text_size], dtype=np.uint8)
            for i, j in product(range(0, len(texts)), range(0, text_size)):
                text = texts[i]
                y_labels[i, j] = alphabet.index(text[j])

            # Generate categorical labels
            y = np.zeros([len(texts), text_size, len(alphabet)], dtype=np.uint8)
            for i, j in product(range(0, len(texts)), range(0, text_size)):
                y[i, j, :] = to_categorical(y_labels[i, j], len(alphabet))

            # Load images & process them
            X = np.zeros([len(texts)] + image_dims, dtype=np.float32)

            for i, image in zip(range(0, len(images)), images):
                x = skimage.io.imread(data_dir + '/' + image)
                # Make sure all images have the correct shape
                if x.shape == image_dims:
                    X[i, :, :, :] = x

            return {
                'X': X,
                'y': y,
                'y_labels': y_labels,
                'alphabet': alphabet
            }


        # Try to reuse preprocessed data.
        preprocessed_data_file = join(global_config.HOME_DIR, '.preprocessed-data.npz')
        try:
            data = dict(np.load(preprocessed_data_file))
        except:
            # Dataset had not be processed yet. Process it and save the results
            # in a file
            data = load_data()
            np.savez_compressed(preprocessed_data_file, **data)

        # Return the preprocessed data
        data = DictNamespace(data)
        return data


    @property
    def num_samples(self):
        '''
        Returns the number of samples on the dataset
        '''
        return self.y.shape[0]

    @property
    def image_dims(self):
        '''
        Returns the size of the images in the dataset. A 3D vector of type:
        height x width x channels
        '''
        return self.X.shape[1:]


    @property
    def X(self):
        '''
        Returns all the images in the dataset as 4D tensor of size:
        n x H x W x C
        Where n is the number of samples, H and W are the width and height of
        the images and C the number of channels for each image
        '''
        return self.data.X


    @property
    def captcha_text_size(self):
        '''
        Returns the number of fixed characters on each captcha image
        '''
        return self.y.shape[1]

    @property
    def alphabet(self):
        '''
        Returns all characters that appears on the captcha images.
        Its a list of string values of size 1
        e.g. 'a', 'b', 'c', ..., 'z'
        The corresponding label or class for each character will be the index
        position that have on this list
        '''
        return self.data.alphabet


    @property
    def y_labels(self):
        '''
        Returns the labels of the samples in the dataset.

        It will be a 2D array of size n x m
        n is the number of samples in the dataset
        m will be the number of characters on each captcha image
        All the values will be in the interval [0, q) where q is the number of
        character classes.
        '''
        return self.data.y_labels

    @property
    def y(self):
        '''
        Returns the categoric labels of the samples in the dataset.

        A 3D array with size n x m x q
        n is the number of samples in the dataset
        m will be the number of characters on each captcha image
        q is the number of character classes.

        It satisfies that:
        for each 0 <= i < n and 0 <= j < m   =>   sum y[i, j, :] is 1
        also for each 0 <= k < q, y[i, j, k] is either 0 or 1 (False or True)

        y_labels is equivalent to np.argmax(y, axis=2)
        '''
        return self.data.y


    def train_test_split(self, test_size=0.15, shuffle=True, random_state=None,
        balance_char_frequencies=True):
        '''
        Performs train/test sets split on this dataset
        Returns two 1D arrays, train_samples, test_samples with the indices of
        the samples that are inside the train or test set respectively

        To get the sample labels for the train set, you can do:
        train_samples, test_samples = dataset.train_test_split()
        y_train_labels = dataset.y_labels[train_samples, :]

        :param test_size: Is the test size normalized in the range (0, 1)
        :param shuffle: Set it to True to shuffle the samples randomly before
        splitting them into train / test sets
        :param random_state: Random seed to shuffle the samples
        :param balance_char_frequencies: If its set to true, it runs an algorithm
        after the train/test split in order to balance the number of occurrences
        of chars between train & test samples
        '''

        # Perform a initial stochastic train/test split
        train_samples, test_samples = sklearn.model_selection.train_test_split(
            range(0, self.num_samples),
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

        if balance_char_frequencies:
            # Balance char frequencies between train & test sets

            y = self.y
            max_iters = 200
            history = np.repeat(np.inf, max_iters).astype(np.float32)
            best_result = None

            # Perform random swaps between samples on train & test to
            # balance char frequencies in both tests
            np.random.seed(random_state)
            for i in range(0, max_iters):
                train_char_f = y[train_samples, :, :].sum(axis=1).mean(axis=0)
                test_char_f = y[test_samples, :, :].sum(axis=1).mean(axis=0)
                loss = np.sum(np.square(train_char_f - test_char_f))
                if np.all(loss < history):
                    best_result = list(train_samples)
                history[i] = loss

                train_rankings = np.maximum(np.multiply(y[train_samples, :, :], train_char_f - test_char_f), 0).sum(axis=2).sum(axis=1)
                test_rankings = np.maximum(np.multiply(y[test_samples, :, :], test_char_f - train_char_f), 0).sum(axis=2).sum(axis=1)
                train_rankings /= train_rankings.sum()
                test_rankings /= test_rankings.sum()

                i = np.nonzero(np.cumsum(train_rankings) >= np.random.rand())[0][0]
                j = np.nonzero(np.cumsum(test_rankings) >= np.random.rand())[0][0]

                test_samples[j], train_samples[i] = train_samples[i], test_samples[j]

            # Now we take the train/test setup that minimizes differencies
            # on char frequencies between both sets
            train_samples = best_result
            test_samples = list(frozenset(range(0, self.num_samples)) - frozenset(train_samples))

            # Now we shuffle again train/test sets
            if shuffle:
                train_samples = sklearn.utils.shuffle(train_samples, random_state=random_state)
                test_samples = sklearn.utils.shuffle(test_samples, random_state=random_state)

        # Returns the result of the train/test split
        return train_samples, test_samples



if __name__ == '__main__':
    import pandas as pd

    dataset = CaptchaDataset()

    print('Loading captcha dataset....')

    df = pd.DataFrame({
        'attr': [
            'Dataset directory',
            'Number of samples',
            'Image dimensions',
            'Number of chars per captcha image',
            'Number of char classes'
        ],
        'values': [
            dataset.config.DATASET_DIR,
            dataset.num_samples,
            dataset.image_dims,
            dataset.captcha_text_size,
            len(dataset.alphabet),
        ]
    })
    df.set_index('attr', drop=True, inplace=True)
    print(df)
