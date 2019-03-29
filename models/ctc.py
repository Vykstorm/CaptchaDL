

import operator
from functools import lru_cache, reduce
from itertools import islice, repeat

from dataset import CaptchaDataset
from input import InputFlow

import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.callbacks import LambdaCallback
from keras.utils import plot_model


def ctc_loss(labels, y_pred, input_length, label_length):
    '''
    CTC loss function
    '''
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)




class CTCInputFlow(InputFlow):
    '''
    This class allows to create batches of samples to train/test our CTC model
    A tuple with 2 items (inputs and outputs) will be returned on each batch
    iteration
    inputs is a dictionary like the next:
    inputs = {
        'labels': array nxm
        'images': array of n gray images (n x image.height x image.width x 1)
        'input_length': array nx1
        'label_length': array nx1
    }
    where n is the batch size

    And outputs is a dictionary with only 1 entry:
    outputs = {
        'CTC': np.zeros([n])
    }

    '''
    def __iter__(self):
        '''
        Returns a iterable object which iterates over all the batches
        and yields tuples of kind (inputs, outputs)
        '''
        it = super().__iter__()
        while True:
            X_batch, y_batch = next(it)
            n, m = X_batch.shape[0], y_batch.shape[1]
            image_height, image_width = X_batch.shape[1:3]
            num_char_classes = y_batch.shape[2]

            labels = y_batch.argmax(axis=2)

            inputs = {
                'labels': labels,
                'images': X_batch,
                'input_length': np.repeat(image_width // 16, n).reshape([-1, 1]),
                'label_length': np.repeat(labels.shape[1], n).reshape([-1, 1])
            }
            outputs = {
                'CTC': np.zeros([n])
            }
            yield inputs, outputs


class CTCModel(Model):
    '''
    Connectionist temporal classifier to predict text inside images
    Inspired by Alex Graves work:
    https://www.cs.toronto.edu/~graves/icml_2006.pdf
    I follow also this guide: https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/#disqus_thread
    '''
    def __init__(self):
        '''
        Constructor
        :param text_size: This indicates the fixed amount of characters to be
        predicted by the classifier on each image.
        :param num_char_classes: This is the number of unique character classes
        to be used when predicting texts
        :param image_dims: The dimensions of the input images. Must be a tuple
        with the next parameters: height x width x n.channels
        '''
        dataset = CaptchaDataset()
        self.text_size = dataset.text_size
        self.num_char_classes = dataset.num_char_classes
        self.image_dims = dataset.image_dims
        image_height, image_width = self.image_dims[0:2]


        # Model inputs
        input_images = Input(shape=self.image_dims, name='images', dtype=np.float32)
        labels = Input(name='labels', shape=(self.text_size,), dtype=np.int64)
        input_length = Input(name='input_length', shape=(1,), dtype=np.int64)
        label_length = Input(name='label_length', shape=(1,), dtype=np.int64)
        inputs = {
            'images': input_images,
            'labels': labels,
            'input_length': input_length,
            'label_length': label_length
        }
        x = input_images

        # Feature extraction
        x = self.features(x)

        # Dimensionallity reduction
        x = Dense(256, activation='relu', kernel_initializer='he_normal', name='dim-reduction')(x)
        x = Dense(128, activation='relu')(x)

        # RNN layers
        x = self.rnn(x)

        # Output layers (softmax)
        x = Dense(self.num_char_classes + 1, activation='softmax', kernel_initializer='he_normal', name='softmax')(x)
        y_pred = x

        # CTC cost
        x = Lambda(lambda args: ctc_loss(*args), output_shape=(1,), name='CTC')([labels, y_pred, input_length, label_length])

        # Real output of the model is the CTC cost
        outputs = [x]

        # Initialize super class
        super().__init__(inputs=list(inputs.values()), outputs=outputs)

        # We define a function which takes the inputs and outputs y_pred
        # We will decode softmax to get the predicted labels by the model
        self.test_func = K.function([input_images], [y_pred])


    def features(self, input_images):
        '''
        This method builts a cnn layers of the model
        It takes as input a 3D tensor (input images) and returns the tensor with
        the images features extracted
        '''
        image_height, image_width = self.image_dims[0:2]

        x = input_images

        # Transpose images
        x = Permute((2, 1, 3))(x)

        # Image feature extraction
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(x)
        x = MaxPool2D((2, 2), name='pool1')(x)

        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPool2D((2, 2), name='pool2')(x)

        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(x)
        x = MaxPool2D((2, 2), name='pool3')(x)

        x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(x)
        x = MaxPool2D((2, 2), name='pool4')(x)

        #x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(x)
        #x = MaxPool2D((2, 2), name='pool4')(x)

        # Merge 2nd and 3th dimension of the tensor
        x = Reshape([image_width // 16, (image_height // 16) * 128], name='image-features')(x)

        return x


    def rnn(self, x):
        '''
        This method builts the rnn layers of the model
        It takes a 3D tensor as input (x) and returns the output of rnn layers
        '''
        #lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm')(x)
        #lstm_2 = LSTM(128, return_sequences=True, kernel_initializer='he_normal', go_backwards=True, name='lstm-2')(x)
        # x = Concatenate()([lstm_1, lstm_2])
        gru_1 = GRU(256, return_sequences=True, name='gru')(x)
        gru_2 = GRU(256, return_sequences=True, go_backwards=True, name='gru2')(x)
        x = Concatenate()([gru_1, gru_2])
        return x


    def compile(self, **kwargs):
        '''
        Compile the model
        '''
        super().compile(loss={'CTC': lambda y_true, y_pred: y_pred}, **kwargs)


    def predict(self, X):
        '''
        Use this model to predict caption text from the images indicated
        Returns a list of items. Each item is the prediction for each image and will be
        a sequence of numeric labels. Each sequence will have a length equal or lower than dataset.text_size
        '''
        y_pred = model.test_func([X])[0]
        out = K.get_value(K.ctc_decode(y_pred, np.repeat(CaptchaDataset().text_size, X.shape[0]), greedy=False, top_paths=1, beam_width=100)[0][0])
        return out


    def predict_text(self, X):
        '''
        Its the same as predict() but it returns a list of strings instead of numeric label sequences
        '''
        return CaptchaDataset().labels_to_text(self.predict(X))


    def plot(self, filename='model.png'):
        '''
        Plot the model and save it on the filename specified as an image
        '''
        plot_model(self, to_file=filename, show_shapes=True, show_layer_names=True)




if __name__ == '__main__':
    model = CTCModel()
    model.predict_text(CaptchaDataset().X[0:4])
