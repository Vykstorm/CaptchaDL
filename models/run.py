

from importlib import import_module
import pandas as pd
import json
import argparse
from shutil import rmtree
from re import match
from inspect import isclass
from os.path import join
from config import global_config


'''
This script provides a command line tool to run our deep learning models
'''


if __name__ == '__main__':
    # Create the cmd parser
    parser = argparse.ArgumentParser(description='Deep learning model CLI to classify captcha images')
    parser.add_argument('model', nargs=1, type=str, help='The model you want to run')
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
    parser.add_argument('--gen-samples', '-g', nargs=1, type=float, metavar='FACTOR',
                        help='This will generate more samples on the dataset (same images but modified using affine transformations) for training the model.' +\
                            'For example, if this value is 2, the train set size will grow up by a factor of 2')
    parser.add_argument('--tensorboard', '-tb', action='store_true', default=False, help='Enable tensorboard logging')
    parser.add_argument('--tensorboard-log-dir', '--tb-log-dir', nargs=1, type=str, metavar='LOG-DIR', default=['.tb-logs'], help='The directory where the tensorboard logs will be stored')

    parsed_args = parser.parse_args()


    # Required imports (we do it after parsing cmd args so that the user response
    # when doing  run --help is minimum)
    from dataset import CaptchaDataset
    from model import Model
    from input import InputFlow
    from keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback


    # We need to know what model to run
    models_config = {}
    with open('models.json', 'r') as fp:
        for entry in json.load(fp):
            names = entry['name']

            result = match('^([^\.]+)\.([^\.]+)$', entry['class'])
            if not result:
                raise Exception('Invalid models.json configuration file')
            module, cls, path = result.group(1), result.group(2), result.group(0)

            for name in names:
                models_config[name] = {'module': module, 'class': cls, 'path': path}

    model_name = parsed_args.model[0]

    if model_name not in models_config:
        parser.error('There is not model named {}'.format(model_name))

    model_config = models_config[model_name]

    try:
        model_cls = getattr(import_module(model_config['module']), model_config['class'])
    except:
        raise Exception('Failed loading {} class'.format(model_config['path']))

    if not isclass(model_cls) or not issubclass(model_cls, Model):
        raise Exception('{} is not a valid model class'.format(model_config['path']))

    # Now we instantiate the model
    model = model_cls()

    # Parse optional  arguments

    train, evaluate = parsed_args.train, parsed_args.eval
    test_size = parsed_args.test_size[0]
    verbose, print_summary = parsed_args.verbose, parsed_args.summary

    save_weights_file = parsed_args.save[0] if parsed_args.save is not None else None
    load_weights_file = parsed_args.load[0] if parsed_args.load is not None else None

    batch_size = parsed_args.batch_size[0]
    epochs = parsed_args.epochs[0]
    gen_samples = parsed_args.gen_samples[0] if parsed_args.gen_samples is not None else None

    tensorboard = parsed_args.tensorboard
    tensorboard_log_dir = parsed_args.tensorboard_log_dir[0]

    #if not train and not evaluate:
    #    parser.error('You need to specify at least --train or --eval options')

    if train:
        if epochs <= 0:
            parser.error('Number of epochs must be a integer great than zero')
        if batch_size <= 0:
            parser.error('Batch size must be an integer greater than zero')

        if gen_samples is not None and gen_samples <= 1:
            parser.error('Gen samples value must be a number greater than one')

        if gen_samples is None:
            gen_samples = 0
        else:
            gen_samples = gen_samples - 1

        # Clean tensorboard logs
        try:
            rmtree(tensorboard_log_dir)
        except:
            pass

    if evaluate:
        if (test_size <= 0 or test_size >= 100):
            parser.error('Test size must be a percentage value. A number in the range (0, 100)')
        test_size = float(test_size) / 100

    # Compile the model
    model.compile()

    # Print the summary of the model
    if print_summary:
        model.summary()


    dataset = CaptchaDataset()
    X, y = dataset.X, dataset.y

    # Load the initial weights
    if load_weights_file:
        try:
            model.load_weights(join(global_config.HOME_DIR, load_weights_file))
        except:
            raise Exception('Failed to load your model weights from file {}'.format(load_weights_file))


    if train:
        # Split data in train / test sets
        train_samples, test_samples = data.train_test_split(test_size=test_size)

        X_train, y_train = X[train_samples], y[train_samples]
        X_test, y_test = X[test_samples], y[test_samples]
    else:
        X_test, y_test = X, y


    if train:
        # Training phase callbacks
        callbacks = [
            # Stop the model if it doesnt improve
            EarlyStopping(monitor='loss', min_delta=0.01, patience=2, verbose=verbose, mode='min'),
        ]
        if verbose:
            # Evaluate test set at the end of each epoch
            callbacks.append(
                LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.evaluate(X_test, y_test, verbose=False)))
            )

        if tensorboard:
            # Enable tensorboard logging
            callbacks.append(
                TensorBoard(log_dir=join(global_config.HOME_DIR, tensorboard_log_dir), write_graph=True, update_freq='batch')
            )

        # Train the model
        print('\nTraining the model...\n')
        model.fit_generator(InputFlow(X_train, y_train, batch_size=batch_size, generate_samples=gen_samples*y_train.shape[0]),
                            verbose=verbose,
                            epochs=epochs,
                            callbacks=callbacks)

    # Save model weights
    if save_weights_file:
        try:
            model.save_weights(join(global_config.HOME_DIR, save_weights_file))
        except:
            raise Exception('Failed to save your model weights to file {}'.format(save_weights_file))

    # Evaluate the model
    if evaluate:
        print('\nEvaluating the model...\n')

        results = model.evaluate(X_test, y_test, verbose=verbose)
        df = pd.DataFrame({'metrics': list(results.keys()), 'values': [round(value, 3) for value in results.values()]  })
        df.set_index('metrics', inplace=True)

        print('\nModel evaluation results:\n')
        print(df)
