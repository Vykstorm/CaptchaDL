

from sklearn import model_selection
import numpy as np

def train_test_split(X, y, test_size=0.15, shuffle=True, random_state=13, stratify=True):
    '''
    Perform train & test set split
    '''
    data = np.load('./preprocessed-data.npz')
    X, y = data['X'], data['y']

    train_indices, test_indices = model_selection.train_test_split(
        np.array(range(0, X.shape[0])), test_size=0.15, shuffle=shuffle, random_state=random_state)


    if stratify:
        np.random.seed(random_state)
        max_iters = 200
        history = np.repeat(np.inf, max_iters).astype(np.float32)
        best_result = None

        for i in range(0, max_iters):
            train_char_f = y[train_indices, :, :].sum(axis=1).mean(axis=0)
            test_char_f = y[test_indices, :, :].sum(axis=1).mean(axis=0)
            loss = np.sum(np.square(train_char_f - test_char_f))

            if np.all(loss < history):
                best_result = {'train':train_indices, 'test':test_indices, 'loss':loss, 'iter':i}
            history[i] = loss

            train_rankings = np.maximum(np.multiply(y[train_indices, :, :], train_char_f - test_char_f), 0).sum(axis=2).sum(axis=1)
            test_rankings = np.maximum(np.multiply(y[test_indices, :, :], test_char_f - train_char_f), 0).sum(axis=2).sum(axis=1)
            train_rankings /= train_rankings.sum()
            test_rankings /= test_rankings.sum()
            i = np.nonzero(np.cumsum(train_rankings) >= np.random.rand())[0][0]
            j = np.nonzero(np.cumsum(test_rankings) >= np.random.rand())[0][0]
            test_indices[j], train_indices[i] = train_indices[i], test_indices[j]

        train_indices = best_result['train']
        test_indices = best_result['test']

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
