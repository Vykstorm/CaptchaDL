
'''
This script creates different ways to compute the score of a model given its
label predictions and the truth labels
'''

import numpy as np
from functools import partial

def char_accuracy(y_true, y_pred):
    '''
    This function counts the number of correct character matches and divides it
    by the total number of characters
    '''
    return (y_pred == y_true).flatten().mean()

def word_accuracy(y_true, y_pred):
    '''
    This function counts the number of fully matched captchas and divides it by
    the number of captchas
    '''
    return np.all(y_pred == y_true, axis=1).mean()


def match_accuracy(n, y_true, y_pred):
    '''
    This function counts the number of partially matched captchas where there are at
    least n occurences and divides it by the number of captchas
    '''
    return ((y_pred == y_true).sum(axis=1) >= n).mean()

match1_accuracy = partial(match_accuracy, 1)
match2_accuracy = partial(match_accuracy, 2)
match3_accuracy = partial(match_accuracy, 3)
match4_accuracy = partial(match_accuracy, 4)
