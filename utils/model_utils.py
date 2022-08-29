from pandas import DataFrame
from numpy import array, arange, reshape, asarray
from constants import *
from pickle import load
from utils import DataUtils
from os.path import exists
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from math import ceil

X_test, y_test = None, None


class ModelUtils:

    @staticmethod
    def get_data_for_time_series_models(target_column: str):
        data = DataUtils.get_country_data()
        train, test = ModelUtils.train_test_split(data[target_column], test_size=0.20)
        return train, test

    @staticmethod
    def get_data_for_linear_model(features: list, target_columns: list, lag_data: bool = False):
        X_train, y_train, X_test, y_test = None, None, None, None
        if not exists('models/linear_data.pkl'):
            data = DataUtils.get_country_data().copy()
            data['TimeUnit'] = arange(data.shape[0])
            data = data.assign(active_cases_lag=data['active_cases'].diff().fillna(0).values,
                               cured_lag=data['cured'].diff().fillna(0).values,
                               death_lag=data['death'].diff().fillna(0).values)
            train, test = ModelUtils.train_test_split(data, test_size=0.25)
            if 'Lag1' in features:
                features.remove('Lag1')
                features.extend([feature + '_lag' for feature in target_columns])
                print(features)
            X_train, y_train = train[features], train[target_columns]
            print(X_train.columns)
            X_test, y_test = test[features], test[target_columns]
            dump({
                'X_train': {'original': train[['TimeUnit', 'active_cases', 'cured', 'death']],
                            'lag1': train[['TimeUnit', 'active_cases_lag', 'cured_lag', 'death_lag']]},
                'X_test': {'original': test[['TimeUnit', 'active_cases', 'cured', 'death']],
                           'lag1': test[['TimeUnit', 'active_cases_lag', 'cured_lag', 'death_lag']]},
                'y_train': y_train,
                'y_test': y_test
            }, open('models/linear_data.pkl', 'wb'))
        else:
            linear_data = load(open('models/linear_data.pkl', 'rb'))
            data_type = 'original' if 'Lag1' not in features else 'lag1'
            X_train, y_train = linear_data['X_train'][data_type], linear_data['y_train']
            X_test, y_test = linear_data['X_test'][data_type], linear_data['y_test']
        return X_train, y_train, X_test, y_test

    @staticmethod
    def get_data_for_memory_based_model():
        sc, X_train, y_train, X_test, y_test = None, None, None, None, None
        if not exists('models/mem_data.pkl'):
            data = DataUtils.get_country_data()[['active_cases', 'cured', 'death']]
            sc = MinMaxScaler(feature_range=(0, 1))
            scaled_data = sc.fit_transform(data.values)
            X, y = ModelUtils.get_XY_from_data(scaled_data, 14)
            X_train, X_test = ModelUtils.train_test_split(X, test_size=0.25)
            y_train, y_test = ModelUtils.train_test_split(y, test_size=0.25)
            X_train = reshape(X_train, (X_train.shape[0], 3, X_train.shape[1]))
            X_test = reshape(X_test, (X_test.shape[0], 3, X_test.shape[1]))
            dump({
                'sc': sc,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }, open('models/mem_data.pkl', 'wb'))
        else:
            mem_data = load(open('models/mem_data.pkl', 'rb'))
            sc = mem_data['sc']
            X_train, y_train = mem_data['X_train'], mem_data['y_train']
            X_test, y_test = mem_data['X_test'], mem_data['y_test']
        return sc, X_train, y_train, X_test, y_test

    @staticmethod
    def get_modified_feature_list(features: list, target: list):
        if 'Lag1' in features:
            features.remove('Lag1')
            features.extend([ feature+'_lag' for feature in target])
        return features

    @staticmethod
    def get_test_data(features: list) -> (DataFrame, DataFrame):
        global X_test
        global y_test
        if X_test is None or y_test is None:
            data = DataUtils.get_country_data().copy()
            data['TimeUnit'] = arange(data.shape[0])
            target_columns = ['cured', 'death', 'active_cases']
            diff = data.drop(columns='Date').diff().fillna(0)
            data = data.assign(lag1_cured=diff['active_cases']) \
                .assign(lag1_death=diff['death']) \
                .assign(lag1_active_cases=diff['active_cases'])
            target_columns.insert(0, 'TimeUnit')
            target_columns.extend(['lag1_cured', 'lag1_death', 'lag1_active_cases'])
            train, test = ModelUtils.train_test_split(data[target_columns], test_size=0.25)
            target_columns.pop(0)
            X_test, y_test = test, test[['cured', 'death', 'active_cases']]
            data = None
            train = None
            test = None
        features = ModelUtils.get_modified_feature_list(features, y_test.columns)
        return X_test[features], y_test

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
    def get_model(model_name=RECURRENT_NEURAL_NETWORK, num_nodes=60, time_steps=15, num_features=1):
        from keras.models import Sequential
        from keras.layers import Input, LSTM, SimpleRNN, Dense, Dropout
        model = Sequential()
        model.add(Input(shape=(num_features, time_steps)))
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

    @staticmethod
    def get_pickled_model(model_name: str, features: list, base_model: str = None,
                          get_tuned_model: bool = False) -> object:
        filename = ''
        if model_name in [SIMPLE_EXPONENTIAL_SMOOTHING, DOUBLE_EXPONENTIAL_SMOOTHING, TRIPLE_EXPONENTIAL_SMOOTHING, ARIMA, SARIMA]:
            filename = 'models/{}_{}.model'.format(model_name, features[0])
        elif base_model is None and features is None:
            filename = 'models/{}{}.model'.format(model_name, '_tuned' if get_tuned_model else '')
        elif base_model is None:
            filename = 'models/{}_[{}].model'.format(model_name, ','.join(features))
        else:
            filename = 'models/{}_{}_[{}].model'.format(model_name, base_model, ','.join(features))
        print('Filename:',filename, ' status:',exists(filename))
        return load(open(filename, 'rb'))
