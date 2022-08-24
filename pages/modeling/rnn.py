#
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.metrics import r2_score, mean_squared_error
from keras.layers import Input, Dense, Dropout, LSTM, SimpleRNN
from keras.models import Model, Sequential
# from keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(Input(shape=(1, 10)))
from utils import DataUtils


def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def get_XY(dat, time_steps):
    # Indices of target array
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    # Prepare X
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))
    return X, Y

if __name__ == '__main__':
    data = DataUtils.get_country_data().copy()
    train_size = 720
    train, test = data['active_cases'].values[:train_size], data['active_cases'].values[train_size:]
    time_steps = 7
    trainX, trainY = get_XY(train, time_steps)
    testX, testY = get_XY(train, time_steps)
    model = keras.models.Sequential([
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            input_shape=[None]),
        keras.layers.SimpleRNN(20, return_sequences=True,
                               input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.Dense(1),
        keras.layers.Lambda(lambda x: x * 100.0)
    ])

    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))
    optimizer = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    model = create_RNN(hidden_units=120, dense_units=1, input_shape=(time_steps,1),
                   activation=['tanh', 'tanh'])
    print(model.summary())
    history = model.fit(trainX, trainY, epochs=100)
    predictions = model.predict(testX)
    print('R2 score:', r2_score(testY, predictions))
    print('MSE:', mean_squared_error(testY, predictions))
    plt.figure()
    sns.lineplot(np.arange(len(testY)), testY, color='blue', label='Actual')
    sns.lineplot(np.arange(len(testY)), np.reshape(predictions, (predictions.shape[0])), color='red', label='Predicted')
    plt.show()