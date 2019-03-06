

import keras
from keras.layers import Input

from dataset import CaptchaDataset
from metrics import *
from input import InputFlow

class Model(keras.models.Model):
    '''
    This is the base class for all our deep learning models
    '''
    def __init__(self):
        dataset = CaptchaDataset()
        t_in = Input(shape=dataset.image_dims)
        t_out = self.build(
            input=t_in,
            num_outputs=dataset.captcha_text_size,
            output_units=len(dataset.alphabet)
            )
        super().__init__(inputs=[t_in], outputs=t_out)


    def fit(self, X, y, **kwargs):
        '''
        Fits the model with the training data.
        '''
        return super().fit(X, [y[:, k, :] for k in range(y.shape[1])], **kwargs)

    def fit_generator(self, generator, steps_per_epoch=None, **kwargs):
        '''
        Fits the model with the training data provided by the given generator
        (an instance of the class InputFlow, generator object or iterator)
        '''
        if not steps_per_epoch and isinstance(generator, InputFlow):
            steps_per_epoch = generator.repetitions.sum() // generator.batch_size
            generator = iter(generator)

        def generator_wrapper():
            while True:
                X_batch, y_batch = next(generator)
                yield (X_batch, [y_batch[:, k, :] for k in range(0, y_batch.shape[1])])

        super().fit_generator(generator_wrapper(), steps_per_epoch, **kwargs)

    def compile(self, **kwargs):
        '''
        Compile this model. This is just like a regular call to keras.models.Model compile()
        method but if the loss is not specified, it will be 'categorical_crossentropy' and
        the default optimizer will be 'rmsprop'
        '''
        if 'loss' not in kwargs:
            kwargs['loss'] = 'categorical_crossentropy'
        if 'optimizer' not in kwargs:
            kwargs['optimizer'] = 'rmsprop'
        return super().compile(**kwargs)


    def predict(self, X):
        '''
        This method uses this model to predict the values for the indicated input samples
        X must be a 4D tensor: n.samples x height x width x n.channels

        Returns A 3D tensor of size n.samples x m x q with the model predictions
        m is the number of different outputs of the model (number of character on each captcha text)
        q is the number of units for each output in the model
        '''
        return np.stack(super().predict(X), axis=1).astype(np.uint8)


    def predict_labels(self, X):
        '''
        This method its the same as predict() but returns a 2D tensor of size n.samples x m

        y_pred_labels = predict_labels(X)

        where y_pred_labels[0, 0] is a number between (0, q)
        q is the number of characters in the alphabet. So that if y_pred_labels[0, 0] is k,
        the predicted char for the first sample at first position is the kth character in the alphabet
        '''
        return np.argmax(self.predict(X), axis=2).astype(np.uint8)


    def predict_text(self, X):
        '''
        This is the same as predict function but this returns a list of strings with the text predictions
        for each captcha image. All the strings have the same length
        '''
        y_labels = self.predict_labels(X)
        alphabet = CaptchaDataset().alphabet
        n = len(alphabet)

        chrs = dict(zip(range(0, n), alphabet))

        y_pred_texts = []
        for i in range(0, y_labels.shape[0]):
            text = ''.join([chrs[label] for label in y_labels[i, :]])
            y_pred_texts.append(text)
        return y_pred_texts


    def evaluate(self, X, y):
        '''
        Use this model to predict the labels for the input samples and compare them
        with the truth labels using different metrics
        '''
        y_pred_labels = self.predict_labels(X)
        y_labels = np.argmax(y, axis=2)

        return {
            'char_accuracy': char_accuracy(y_pred_labels, y_labels),
            'word_accuracy': word_accuracy(y_pred_labels, y_labels),
            'match1_accuracy': match1_accuracy(y_pred_labels, y_labels),
            'match2_accuracy': match2_accuracy(y_pred_labels, y_labels),
            'match3_accuracy': match3_accuracy(y_pred_labels, y_labels),
            'match4_accuracy': match4_accuracy(y_pred_labels, y_labels)
        }


    def build(self, input, output_units, num_outputs):
        '''
        This method must build the model layers and return a list of tensors which corresponds
        to the different outputs that the model must generate
        This must be implemented by subclasses
        :param input: This must be the input layer of the model
        :param output_units: Each output of the model must have amount of units
        :param num_outputs: The model must generate this amount of outputs

        :return: This method must return a list of output tensors. Each of them corresponding with
        different model outputs
        '''
        raise NotImplementedError()
