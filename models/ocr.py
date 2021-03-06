


import numpy as np
from dataset import CaptchaDataset
from chars import find_chars
from char_classifier import CharClassifier


class OCRModel:
    '''
    This class is a estimator that takes captcha images and predicts the text inside
    them.
    First it extracts the individual characters from the image

    Then characters are classified one by one using a convolutional neuronal network
    (generated by the script char_classifier.py)
    '''
    def __init__(self):
        '''
        Initializes this model
        :param text_size: Must be the number of characters to predict for each sample image
        '''
        self.char_classifier = CharClassifier()
        # Load pretrained weights
        self.char_classifier.load_weights()

    def predict(self, X):
        '''
        Use this model to predict the texts inside the images specified
        :param X: Must be a 4D array of size: n.samples x img.height x img.width x 1

        :return A 3D array 'y' of size n.samples x text-size x alphabet-size
        Where the element y[i, j, k] indicates the probability of the jth character on
        the ith image to be the label k
        '''
        dataset = CaptchaDataset()
        num_classes, text_size = dataset.num_char_classes, dataset.text_size
        char_size = self.char_classifier.layers[0].input_shape[1:3]

        y = np.zeros([X.shape[0], text_size, num_classes]).astype(np.float32)

        for i in range(0, X.shape[0]):
            chars = find_chars(X[i, :, :, 0], char_size, num_chars=text_size).reshape((-1,) + char_size + (1,))
            y[i] = self.char_classifier.predict(chars)

        return y

    def predict_labels(self, X):
        '''
        Its the same as self.predict(X).argmax(axis=2)
        :param X: Must be a 4D array of size: n.samples x img.height x img.width x 1
        :return A 2D array 'y_labels' of size n.samples x text-size of integer values in the range [0, alphabet-size)
        The value y_labels[i, j] indicates the predicted label for the jth character on the
        ith image
        '''
        return self.predict(X).argmax(axis=2)

    def predict_text(self, X):
        '''
        Its the same as predict_abels() but this returns directly a char values instead of integer labels
        :return A 2D array of size n.samples x text-size of char values
        '''
        return CaptchaDataset().labels_to_text(self.predict_labels(X))


if __name__ == '__main__':
    from input import InputFlow
    import matplotlib.pyplot as plt
    from metrics import summary

    # Get dataset images
    dataset = CaptchaDataset()
    X, y = dataset.X, dataset.y


    # Build the model
    model = OCRModel()


    # The next lines will show a bunch of captcha images & the predictions made by
    # the model

    indices = np.random.choice(np.arange(0, dataset.num_samples), size=9)
    X_batch, y_batch = next(iter(InputFlow(X, y, batch_size=9)))

    # Predict texts inside images
    texts = [''.join([char.item().decode() for char in text]) for text in dataset.labels_to_text(y_batch.argmax(axis=2))]
    texts_pred = [''.join([char.item().decode() for char in text]) for text in model.predict_text(X_batch)]


    rows, cols = X_batch.shape[0] // 3, 3

    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
    for i in range(0, rows):
        for j in range(0, cols):
            k = i * cols + j
            plt.sca(ax[i, j])

            plt.imshow(X_batch[k, :, :, 0], cmap='gray')
            plt.title('Labels: "{}", Prediction: "{}"'.format(texts[k], texts_pred[k]))

            plt.xticks([])
            plt.yticks([])
    plt.suptitle('Batch of image samples')
    plt.tight_layout()



    # Now evaluate the model on a test set & show metric scores
    X_test, y_test = next(iter(InputFlow(X, y, batch_size=1000)))

    print('Predicting captcha text images...')
    print('Number of images: {}'.format(X_test.shape[0]))
    y_labels = y.argmax(axis=2)
    y_pred_labels = model.predict_labels(X)

    # Show evaluation summary
    print('Evaluation summary:')
    summary(y_labels, y_pred_labels)

    plt.show()
