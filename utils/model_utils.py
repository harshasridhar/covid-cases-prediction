from pandas import DataFrame
from numpy import array
from constants import *

class ModelUtils:

    @staticmethod
    def train_test_split(data: DataFrame, test_size: float) -> (DataFrame, DataFrame):
        train_index = int(data.shape[0] * (1 - test_size))
        return data[:train_index], data[train_index:]

    @staticmethod
    def get_XY_from_data(data: DataFrame, time_steps: int):
        X, y = [], []
        for i in range(data.shape[0] - time_steps):
            x = data[i: i + time_steps]
            X.append(x)
            y.append(data[i + time_steps])
        return array(X), array(y)

    @staticmethod
    def get_model(model_name=RECURRENT_NEURAL_NETWORK, num_nodes=60, time_steps=15):
        from keras.models import Sequential
        from keras.layers import Input, LSTM, SimpleRNN, Dense, Dropout
        model = Sequential()
        model.add(Input(shape=(1, time_steps)))
        if RECURRENT_NEURAL_NETWORK == model_name:
            model.add(SimpleRNN(num_nodes, return_sequences=True))
        else:
            model.add(LSTM(num_nodes, return_sequences=True))
        model.add(Dropout(0.4))
        if RECURRENT_NEURAL_NETWORK == model_name:
            model.add(SimpleRNN(num_nodes, return_sequences=True))
        else:
            model.add(LSTM(num_nodes, return_sequences=True))
        model.add(Dropout(0.2))
        if RECURRENT_NEURAL_NETWORK == model_name:
            model.add(SimpleRNN(num_nodes))
        else:
            model.add(LSTM(num_nodes))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu'))
        return model

    @staticmethod
    def r_square(y_true, y_pred):
        from keras import backend as K
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))