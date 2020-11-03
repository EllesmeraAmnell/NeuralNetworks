import numpy as np
from keras import models
from keras import layers
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from plot import plot_history
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

number_of_features = 4
configuration = 2
new_model = True


def create_network():
    network = models.Sequential()
    network.add(layers.Dense(units=4, activation='relu', input_shape=(number_of_features,)))
    network.add(layers.Dense(units=10, activation='relu'))
    # network.add(layers.Dense(units=20, activation='relu'))
    network.add(layers.Dense(units=3, activation='sigmoid'))

    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return network


def prepare_data(from_file=True):
    iris = load_iris()
    x = iris.data
    y = iris.target

    if not from_file:
        # Split and save to file
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)  # 20% - test
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75,
                                                          random_state=1)  # 60% - train, 20% - val
        x_train.tofile('x_train.txt')
        x_test.tofile('x_test.txt')
        x_val.tofile('x_val.txt')
        y_train.tofile('y_train.txt', sep=';')
        y_test.tofile('y_test.txt', sep=';')
        y_val.tofile('y_val.txt', sep=';')
    else:
        # From file
        x_train = np.fromfile('x_train.txt').reshape(90, 4)
        y_train = np.fromfile('y_train.txt', 'int32', sep=';')
        x_test = np.fromfile('x_test.txt').reshape(30, 4)
        y_test = np.fromfile('y_test.txt', 'int32', sep=';')
        x_val = np.fromfile('x_val.txt').reshape(30, 4)
        y_val = np.fromfile('y_val.txt', 'int32', sep=';')

    # Coding to [1, 0, 0], [0, 1, 0], [0, 0, 1]
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test, x_val, y_val


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val = prepare_data()

    epochs = 300
    path = f'{configuration}.mdl_wts.hdf5'

    if new_model:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(path, save_best_only=True, monitor='val_loss', mode='min')

        model = create_network()
        final_model = model.fit(x_train, y_train, epochs=epochs, verbose=0,
                                callbacks=[earlyStopping, mcp_save], validation_data=(x_val, y_val))

        stopped_epoch = earlyStopping.stopped_epoch if earlyStopping.stopped_epoch != 0 else epochs
        print(f'Stopped epoch: {stopped_epoch}')

        plot_history(final_model, f'{configuration}')
        scores = model.evaluate(x_test, y_test, verbose=0)
        print(f'Mean accuracy %: {100 * scores[1]}')

    else:
        model = load_model(path)
        new_scores = model.evaluate(x_test, y_test, verbose=0)
        print(f'Mean accuracy %: {100 * new_scores[1]}')
