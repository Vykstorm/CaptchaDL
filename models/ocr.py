


from model import Model
from keras.layers import *

class OCRModel(Model):
    '''
    Very basic OCR model that predicts the captchas text.
    '''

    def build(self, input, output_units, num_outputs):

        x = input

        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(x)
        x = MaxPool2D((2, 2), name='pooling1')(x)
        x = Dropout(0.4, name='dropout1')(x)

        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPool2D((2, 2), name='pooling2')(x)
        x = Dropout(0.4, name='dropout2')(x)

        x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(x)
        x = MaxPool2D((2, 2), name='pooling3')(x)
        x = Dropout(0.4, name='dropout3')(x)


        t_lstm_in = x
        t_out = []

        for k in range(0, num_outputs):
            x = Lambda(lambda x: x[:, :, k*5:(k+1)*5,:])(t_lstm_in)
            x = Reshape((1, -1))(t_lstm_in)
            x = LSTM(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(output_units, activation='softmax', name='output_{}'.format(k))(x)
            t_out.append(x)


        return t_out
