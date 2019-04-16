
'''
This script creates different ways to compute the score of a model given its
label predictions and the truth labels
'''

import numpy as np
from functools import partial, update_wrapper
from keras.callbacks import BaseLogger
from json import JSONEncoder
import keras.backend as K
import pandas as pd


'''
The next functions will have always the same signature. They take the truth and
predicted labels and compare them. Both must be a 2D tensor of int64 values with
the same size (nxm).
n is interpreted as the number of samples classified
m is the number of labels on each sample

Value -1 in the predicted labels will represent a 'blank' space character (null character)

'''


def metric(f):
    '''
    This is a decorator for all metric functions
    '''
    def wrapper(y_true, y_pred, categorical_labels=False, *args, **kwargs):
        y_true, y_pred = K.cast(y_true, np.int64), K.cast(y_pred, np.int64)

        if categorical_labels:
            y_true, y_pred = K.argmax(y_true, axis=2), K.argmax(y_pred, axis=2)

        return f(y_true, y_pred, *args, **kwargs)

    update_wrapper(wrapper, f)
    return wrapper


@metric
def char_accuracy(y_true, y_pred):
    '''
    This metric return the mean of characters matched correctly in total
    '''
    return K.mean(K.cast(K.flatten(K.equal(y_true, y_pred)), np.float32))

@metric
def matchk_accuracy(y_true, y_pred, k=2):
    '''
    This metric returns the mean of sample predictions that at least matches k labels correctly
    k must be a number in the range [1, m] where m is the number of labels on each sample
    '''
    return K.mean(K.cast(K.greater_equal(K.sum(K.cast(K.equal(y_true, y_pred), np.int64), axis=1), k), np.float32))

@metric
def fullmatch_accuracy(y_true, y_pred):
    '''
    This metric returns the mean of sample predictions that matches all the labels correctly
    '''
    return K.mean(K.prod(K.cast(K.equal(y_true, y_pred), np.float32), axis=1))



def summary(y_true, y_pred, categorical_labels=False):
    '''
    Prints on stdout different metrics comparing the truth and
    predicted labels specified as arguments
    '''

    metrics = {
        'char_acc': char_accuracy(y_true, y_pred, categorical_labels),
        'fullmatch_acc': fullmatch_accuracy(y_true, y_pred, categorical_labels)
    }
    for k in range(1, y_true.shape[1]):
        metrics['match{}_acc'.format(k)] = matchk_accuracy(y_true, y_pred, categorical_labels, k=k)

    df = pd.DataFrame.from_dict(
        dict([(metric, [round(K.get_value(value), 5)]) for metric, value in metrics.items()] + [('-', 'values')])
    )
    df.set_index(['-'], inplace=True)

    print('Number of samples: {}, Number of characters per sample: {}'.format(*y_true.shape))
    print(df)


class FloydhubKerasCallback(BaseLogger):
    '''
    This class can be used as a callback object that can be passed to the method fit()
    when training your model (inside 'callbacks' argument)
    If it is used while your model is running on a floydhub server, training metrics
    will be plotted at real time under the 'Training metrics' panel.
    '''
    def __init__(self, mode='epoch', metrics=None, stateful_metrics=None):
        super().__init__(stateful_metrics)

        if mode not in ('epoch', 'batch'):
            raise ValueError('Mode parameter should be "epoch" or "batch"')

        if metrics is not None and not isinstance(metrics, (list, tuple)):
            raise ValueError('Metrics parameter should be a list of training metric names to track')

        if  stateful_metrics is not None and not isinstance(metrics, (list, tuple)):
            raise ValueError('Stateful metrics parameter should be a list of training metric names to track')

        self.mode = mode
        self.metrics = frozenset(metrics) if metrics is not None else None
        self.encoder = JSONEncoder()

    def report(self, metric, value, **kwargs):
        info = {'metric': metric, 'value': value}
        info.update(kwargs)
        print(self.encoder.encode(info))

    def on_batch_end(self, batch, logs):
        if not self.mode == 'batch':
            return

        metrics = frozenset(logs.keys()) - frozenset(['batch', 'size'])
        if self.metrics:
            metrics &= self.metrics
        for metric in metrics:
            self.report(metric, round(logs[metric].item(), 5), step=batch)

    def on_epoch_end(self, epoch, logs):
        if not self.mode == 'epoch':
            return

        metrics = frozenset(logs.keys())
        if self.metrics:
            metrics &= self.metrics
        for metric in metrics:
            self.report(metric, round(logs[metric].item(), 5), step=epoch)



if __name__ == '__main__':
    '''
    Module unit test
    '''
    import numpy as np
    import unittest
    from unittest import TestCase

    class MetricsUnitCase(TestCase):
        def test_char_accuracy(self):
            '''
            Test char_accuracy metric
            '''
            y_true = np.array([
                [0, 0, 1, 0],
                [1, 0, 1, 1]],
            dtype=np.int64)

            y_pred = np.array([
                [0, 1, 1, 0],
                [-1, 1, 1, 1]],
            dtype=np.int64)

            self.assertEqual(K.get_value(char_accuracy(y_true, y_pred)), 5/8)


        def test_fullmatch_accuracy(self):
            '''
            Test fullmatch accuracy metric
            '''
            y_true = np.array([
                [0, 1, 1, 0],
                [0, 0, 1, 0]],
            dtype=np.int64)

            y_pred = np.array([
                [0, 1, 1, 0],
                [0, 0, 0, 1]],
            dtype=np.int64)

            self.assertEqual(K.get_value(fullmatch_accuracy(y_true, y_pred)), 0.5)


        def test_matchk_accuracy(self):
            '''
            Test matchk accuracy metric
            '''
            y_true = np.array([
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [0, 0, 1, 1]],
            dtype=np.int64)

            y_pred = np.array([
                [0, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 0, 0, 0]],
            dtype=np.int64)

            self.assertEqual(K.get_value(matchk_accuracy(y_true, y_pred, k=1)), 1)
            self.assertAlmostEqual(K.get_value(matchk_accuracy(y_true, y_pred, k=2)), 2/3, delta=0.01)
            self.assertAlmostEqual(K.get_value(matchk_accuracy(y_true, y_pred, k=3)), 1/3, delta=0.01)

    y_true = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1]],
    dtype=np.int64)

    y_pred = np.array([
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 0]],
    dtype=np.int64)

    unittest.main()
