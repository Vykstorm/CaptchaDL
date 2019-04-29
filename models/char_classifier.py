

import numpy as np
import keras
import keras.backend as K
from keras.layers import *

from dataset import CaptchaDataset


class CharClassifier(keras.models.Model):
    '''
    This class defines a convolutional neuronal network to classify individual
    characters on captcha image
    '''
    def __init__(self, char_size=(40, 40)):
        dataset = CaptchaDataset()
        num_classes = dataset.num_char_classes

        # The next lines defines the layers of the CNN

        t_in = Input(shape=char_size + (1,), dtype=np.float32)

        x = t_in

        x = Conv2D(32, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)

        t_out = x

        # Initialize super instance (a keras model)
        super().__init__([t_in], [t_out])

        # Compile the model
        self.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def load_weights(self):
        '''
        Load the weights obtained at the training phase of this model by previous executions
        of this script. If no weights were generated previously, this method dont do anything.
        '''
        try:
            super().load_weights('.char-classifier-weights.hdf5')
        except:
            pass

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import matplotlib.pyplot as plt
    import seaborn as sns
    from argparse import ArgumentParser

    # Process command line arguments
    parser = ArgumentParser(description='CLI to train/evaluate char classifier')
    parser.add_argument('-t', '--train', action='store_true', default=False, help='Train the model')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='Evaluate the model')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbosity mode')
    parser.add_argument('--batch-size', type=int, nargs=1, default=[8], help='Batch size when training the model')
    parser.add_argument('--epochs', type=int, nargs=1, default=[4], help='Max number of epochs of training phase')
    parser.add_argument('--test-size', type=float, nargs=1, default=[0.15], help='Test set size ratio in the range (0,1). Only used when test is enabled')
    parser.add_argument('--num-samples', type=int, nargs=1, help='Number of samples to be used to train/eval the model instead of the whole dataset')

    params = parser.parse_args()


    # Load data generated using gen_char_dataset()
    data = np.load('.chars.npz')
    X, y = data['X'], data['y']
    n = X.shape[0]

    train, eval, verbose = params.train, params.eval, params.verbose
    batch_size, epochs, test_size = params.batch_size[0], params.epochs[0],params.test_size[0]
    num_samples = min(params.num_samples[0], n) if params.num_samples is not None else n

    if num_samples <= 0:
        parser.error('Num samples must be a number greater than 0')

    if not train and not eval:
        parser.error('Either train or eval parameter must be set to true')

    if num_samples < n:
        # Use only part of the dataset
        indices = np.random.choice(np.arange(0, n), size=num_samples, replace=False)
        X, y = X[indices], y[indices]

    # Get char labels
    y_labels = y.argmax(axis=1)

    # Build the model
    model = CharClassifier(X.shape[1:3])

    # Show model info
    if verbose:
        model.summary()

    # Split data into train & test sets
    if train and eval:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, stratify=y_labels)
    elif not eval:
        X_train, y_train = X, y
    else:
        X_test, y_test = X, y

    if train:
        if verbose:
            print('Training the model...')
        # Train the model
        callbacks = [
            EarlyStopping(min_delta=0.01, monitor='val_loss', mode='min', patience=1),
            ModelCheckpoint('.char-classifier-weights.hdf5', monitor='val_loss', mode='min', period=1, save_weights_only=True)
        ]
        result = model.fit(X_train, y_train, batch_size=8, epochs=epochs, verbose=verbose, validation_split=0.1, callbacks=callbacks)
        history = result.history

        # Show train performance score history

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        plt.sca(ax[0])
        plt.plot(history['loss'], color='red')
        plt.plot(history['val_loss'], color='blue')
        plt.legend(['Loss', 'Val. Loss'])
        plt.xlabel('Epoch')
        plt.title('Loss')
        plt.tight_layout()

        plt.sca(ax[1])
        plt.plot(history['acc'], color='red')
        plt.plot(history['val_acc'], color='blue')
        plt.legend(['Accuracy', 'Val. Accuracy'])
        plt.xlabel('Epoch')
        plt.title('Accuracy')

        plt.suptitle('Model performance on training')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

    elif eval:
        # Load previously computed weights if train is off and eval enabled
        model.load_weights()


    if eval:
        if verbose:
            print('Testing the model...')

        # Evaluate the model on test
        y_test_pred = model.predict(X_test, verbose=verbose)

        # Show accuracy score
        y_test_labels = y_test.argmax(axis=1)
        y_test_labels_pred = y_test_pred.argmax(axis=1)
        print('Accuracy on test set: {}'.format(np.round(accuracy_score(y_test_labels, y_test_labels_pred), 4)))

        # Show evaluation confusion matrx
        plt.figure(figsize=(6, 5))
        alphabet = CaptchaDataset().alphabet
        sns.heatmap(confusion_matrix(y_test_labels, y_test_labels_pred), annot=True, fmt='d',
                                    xticklabels=alphabet, yticklabels=alphabet)
        plt.title('Confusion matrix of eval predictions')


    plt.show()
