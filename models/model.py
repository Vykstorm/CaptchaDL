

import keras
from keras.layers import Input
from keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback
import pandas as pd

import argparse

from dataset import CaptchaDataset
from metrics import *
from input import InputFlow


class Model(keras.models.Model):
    '''
    This is the base class for all our deep learning models
    '''
    def __init__(self):
        dataset = CaptchaDataset()
        t_in = Input(shape=dataset.image_dims, name='input')
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
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['accuracy']
        return super().compile(**kwargs)


    def predict(self, X, **kwargs):
        '''
        This method uses this model to predict the values for the indicated input samples
        X must be a 4D tensor: n.samples x height x width x n.channels

        Returns A 3D tensor of size n.samples x m x q with the model predictions
        m is the number of different outputs of the model (number of character on each captcha text)
        q is the number of units for each output in the model
        '''
        return np.stack(super().predict(X, **kwargs), axis=1).astype(np.uint8)


    def predict_labels(self, X, **kwargs):
        '''
        This method its the same as predict() but returns a 2D tensor of size n.samples x m

        y_pred_labels = predict_labels(X)

        where y_pred_labels[0, 0] is a number between (0, q)
        q is the number of characters in the alphabet. So that if y_pred_labels[0, 0] is k,
        the predicted char for the first sample at first position is the kth character in the alphabet
        '''
        return np.argmax(self.predict(X, **kwargs), axis=2).astype(np.uint8)


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


    def evaluate(self, X, y, **kwargs):
        '''
        Use this model to predict the labels for the input samples and compare them
        with the truth labels using different metrics
        '''
        y_pred_labels = self.predict_labels(X, **kwargs)
        y_labels = np.argmax(y, axis=2)

        return {
            'char_accuracy': char_accuracy(y_labels, y_pred_labels),
            'word_accuracy': word_accuracy(y_labels, y_pred_labels),
            'match1_accuracy': match1_accuracy(y_labels, y_pred_labels),
            'match2_accuracy': match2_accuracy(y_labels, y_pred_labels),
            'match3_accuracy': match3_accuracy(y_labels, y_pred_labels),
            'match4_accuracy': match4_accuracy(y_labels, y_pred_labels)
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


    def run(self):
        from dataset import CaptchaDataset
        from input import InputFlow


        # Create the cmd parser
        parser = argparse.ArgumentParser(description='Deep learning model CLI to classify captcha images')
        parser.add_argument('--train', '-t', action='store_true', default=False, help='Train your model')
        parser.add_argument('--batch-size', '-bs', nargs=1, default=[32], type=int, help='Batch size for the training phase')
        parser.add_argument('--epochs', '--iters', nargs=1, default=[10], type=int, help='Number of iterations of the training phase')
        parser.add_argument('--eval', '--test', '-e', action='store_true', default=False,
                            help='Evaluate your model. If training option is enabled, the model will be evaluated on the test set.' +\
                            'If training option is disabled, the model will be evaluated using the whole dataset')
        parser.add_argument('--test-size', nargs=1, default=[15], type=float, help='Percentage of the dataset to be used as test set')
        parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Enable debugging info printing into stdout')
        parser.add_argument('--save', '-s', nargs=1, type=str, metavar='FILE', help='Saves the weights of the model to a file. This will always happen after training the model')
        parser.add_argument('--load', '-i', nargs=1, type=str, metavar='FILE', help='Initialize the weights of the model using the file. This will happen before training the model')
        parser.add_argument('--summary', '-ps', '--print-summary', action='store_true', default=False, help='Print model keras summary')
        parser.add_argument('--gen-samples', '-g', nargs=1, type=float, metavar='FACTOR', default=0,
                            help='This will generate more samples on the dataset (same images but modified using affine transformations) for training the model.' +\
                                'For example, if this value is 2, the train set size will grow by a factor of 2')

        # Parse the arguments
        parsed_args = parser.parse_args()

        train, evaluate = parsed_args.train, parsed_args.eval
        test_size = parsed_args.test_size[0]
        verbose, print_summary = parsed_args.verbose, parsed_args.summary

        save_weights_file = parsed_args.save[0] if parsed_args.save is not None else None
        load_weights_file = parsed_args.load[0] if parsed_args.load is not None else None

        batch_size = parsed_args.batch_size[0]
        epochs = parsed_args.epochs[0]

        if not train and not evaluate:
            parser.error('You need to specify at least --train or --eval options')

        if train:
            if epochs <= 0:
                parser.error('Number of epochs must be a integer great than zero')
            if batch_size <= 0:
                parser.error('Batch size must be an integer greater than zero')

        if evaluate:
            if (test_size <= 0 or test_size >= 100):
                parser.error('Test size must be a percentage value. A number in the range (0, 100)')
            test_size = float(test_size) / 100


        # Print the summary of the model
        if print_summary:
            self.summary()

        # Compile the model
        self.compile()

        # Load the initial weights
        if load_weights_file:
            try:
                self.load_weights(load_weights_file)
            except:
                raise Exception('Failed to load your model weights from file {}'.format(load_weights_file))


        data = CaptchaDataset()
        X, y = data.X, data.y

        if train:
            # Split data in train / test sets
            train_samples, test_samples = data.train_test_split(test_size=test_size)

            X_train, y_train = X[train_samples], y[train_samples]
            X_test, y_test = X[test_samples], y[test_samples]
        else:
            X_test, y_test = X, y


        if train:
            callbacks = [
                EarlyStopping(monitor='loss', min_delta=0.01, patience=2, verbose=verbose, mode='min'),
                LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(self.evaluate(X_test, y_test, verbose=False)))
            ]

            # Train the model
            print('\nTraining the model...\n')
            self.fit_generator(InputFlow(X_train, y_train, batch_size=batch_size),
                                verbose=verbose,
                                epochs=epochs,
                                callbacks=callbacks)

        # Save model weights
        if save_weights_file:
            try:
                self.save_weights(save_weights_file)
            except:
                raise Exception('Failed to save your model weights to file {}'.format(save_weights_file))

        # Evaluate the model
        if evaluate:
            print('\nEvaluating the model...\n')

            results = self.evaluate(X_test, y_test, verbose=verbose)
            df = pd.DataFrame({'metrics': list(results.keys()), 'values': [round(value, 3) for value in results.values()]  })
            df.set_index('metrics', inplace=True)

            print('\nModel evaluation results:\n')
            print(df)
