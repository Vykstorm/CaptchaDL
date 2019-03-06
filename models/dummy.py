


from model import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

class DummyModel(Model):
    '''
    This creates a model that predicts the captchas text
    with a very simple deep learning model
    This is just to show how to build a model
    '''

    def build(self, input, output_units, num_outputs):
        x = Conv2D(36, (3, 3), activation='relu')(input)
        x = MaxPool2D((4, 4))(x)

        x = Conv2D(20, (3, 3), activation='relu')(x)
        x = MaxPool2D((4, 4))(x)
        x = Flatten()(x)

        x = Dense(40, activation='relu')(x)
        x = Dense(output_units, activation='softmax')(x)

        return [x] * num_outputs



if __name__ == '__main__':
    from dataset import CaptchaDataset
    from input import InputFlow

    dataset = CaptchaDataset()
    input = InputFlow(dataset.X, dataset.y, batch_size=16)

    model = DummyModel()
    model.compile()
    #model.summary()
    model.fit_generator(input, verbose=True, epochs=1)
    print(model.evaluate(dataset.X, dataset.y))
