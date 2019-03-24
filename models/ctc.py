

from functools import lru_cache
from itertools import islice

from dataset import CaptchaDataset
from input import InputFlow

import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.callbacks import LambdaCallback

def ctc_lambda_func(args):
    '''
    Helper method to compute CTC loss
    '''
    y_pred, labels, input_length, label_length = args
    #y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def Function(inputs, output):
    '''
    This is a wrapper over keras.backend.function. It defines a function with multiple
    tensor inputs and only one tensor output.
    :param inputs: Must be a list of tensors. It also can be a dictionary where the values
    are all tensors.
    :param output: Must be a tensor

    :return: Returns the generated function
    '''
    if isinstance(inputs, dict):
        f = Function(list(inputs.values()), output)
        return lambda x: f([x[input_name] for input_name in inputs])

    f = K.function(inputs, [output])
    return lambda x: f(x)[0]



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

            labels = y_batch.argmax(axis=2)

            inputs = {
                'labels': labels,
                'images': X_batch,
                'input_length': np.repeat(25, n).reshape([-1, 1]),
                'label_length': np.repeat(m, n).reshape([-1, 1])
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
    def __init__(self, text_size, num_char_classes, image_dims):
        '''
        Constructor
        :param text_size: This indicates the fixed amount of characters to be
        predicted by the classifier on each image.
        :param num_char_classes: This is the number of unique character classes
        to be used when predicting texts
        :param image_dims: The dimensions of the input images. Must be a tuple
        with the next parameters: height x width x n.channels
        '''
        self.text_size = text_size
        self.num_char_classes = num_char_classes
        self.image_dims = image_dims
        image_height, image_width = self.image_dims[0:2]


        # Model inputs
        input_images = Input(shape=image_dims, name='images', dtype=np.float32)
        labels = Input(name='labels', shape=(text_size,), dtype=np.int64)
        input_length = Input(name='input_length', shape=(1,), dtype=np.int64)
        label_length = Input(name='label_length', shape=(1,), dtype=np.int64)
        inputs = {
            'images': input_images,
            'labels': labels,
            'input_length': input_length,
            'label_length': label_length
        }

        x = input_images

        # Image feature extraction
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2))(x)

        # Dimensionallity reduction
        x = Permute((2, 1, 3))(x)
        x = Reshape([image_width // 8, (image_height // 8) * 32])(x)
        x = Dense(32, activation='relu')(x)

        # LSTM net
        x = LSTM(128, return_sequences=True)(x)

        # Output layers (softmax)
        x = Dense(num_char_classes + 1, activation='softmax')(x)
        y_pred = x

        # We define a function which takes the inputs and outputs y_pred
        # Layer we will decode softmax to get the predicted labels by the model
        self.test_func = Function(inputs, y_pred)

        # CTC cost
        ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='CTC')([y_pred, labels, input_length, label_length])

        # Real output of the model is the CTC cost
        outputs = [ctc_loss]

        # Initialize super class
        super().__init__(inputs=list(inputs.values()), outputs=outputs)


    def compile(self, **kwargs):
        '''
        Compile the model
        '''
        super().compile(loss={'CTC': lambda y_true, y_pred: y_pred}, **kwargs)



if __name__ == '__main__':
    K.clear_session()

    # Build the model
    model = CTCModel(text_size=5, num_char_classes=19, image_dims=(50, 200, 1))
    model.summary()

    model.compile(optimizer='rmsprop')
    
