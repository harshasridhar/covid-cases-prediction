import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from utils import DataUtils, ModelUtils
from functools import partial
from numpy import arange, reshape
from os.path import exists
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import r2_score, mean_absolute_error as mae, mean_squared_error as mse
from constants import *
from pickle import dump, load
from time import time
from pandas import DataFrame

data = DataUtils.get_country_data().copy()
data['TimeUnit'] = arange(data.shape[0])
stats = DataFrame(columns=['Model', 'Base', 'Features', 'R2_Score', 'MAE', 'MSE'])
index = 0


def run_model(model_name: str, features: list, base_model: str = None, save_model: bool = False):
    filename = ''
    if base_model is None:
        filename = 'models/{}_[{}].model'.format(model_name, ','.join(features))
    else:
        filename = 'models/{}_{}_[{}].model'.format(model_name, base_model, ','.join(features))
    model = None
    if model_name in [MULTI_OUTPUT_REGRESSOR, CHAINED_REGRESSOR]:
        base_model_obj = LinearRegression() if base_model == LINEAR_REGRESSION else SVR()
        model = MultiOutputRegressor(base_model_obj) if model_name == MULTI_OUTPUT_REGRESSOR else RegressorChain(
            base_model_obj)
    target_columns = ['cured', 'death', 'active_cases']
    X_train, y_train, X_test, y_test = ModelUtils.get_data_for_linear_model(features,target_columns)
    start_time = time()
    model.fit(X_train, y_train)
    time_taken = time() - start_time
    predictions = model.predict(X_test)
    r2 = list(map(partial(round, ndigits=3), r2_score(y_test, predictions, multioutput='raw_values')))
    MAE = list(map(partial(round, ndigits=3), mae(y_test, predictions, multioutput='raw_values')))
    MSE = list(map(partial(round, ndigits=3), mse(y_test, predictions, multioutput='raw_values')))
    # plt.figure()
    # sns.lineplot(arange(len(y_test)), y_test['active_cases'].values, color='blue', label='Actual')
    # sns.lineplot(arange(len(y_test)), [row[2] for row in predictions], color='red', label='Predicted')
    # plt.show()
    global index
    stats.loc[index] = [model_name, base_model, ','.join(features), r2, MAE, MSE]
    index += 1
    print(filename)
    print(r2)
    if save_model:
        dump({'model': model,
              'time_taken': time_taken,
              'r2_score': r2,
              'mse': MSE,
              'mae': MAE
              },
             open(filename, 'wb'))


run_model(MULTI_OUTPUT_REGRESSOR, ['TimeUnit', 'Lag1'], LINEAR_REGRESSION, True)
run_model(MULTI_OUTPUT_REGRESSOR, ['TimeUnit'], LINEAR_REGRESSION, True)
run_model(MULTI_OUTPUT_REGRESSOR, ['Lag1'], LINEAR_REGRESSION, True)
run_model(CHAINED_REGRESSOR, ['TimeUnit', 'Lag1'], LINEAR_REGRESSION, True)
run_model(CHAINED_REGRESSOR, ['TimeUnit'], LINEAR_REGRESSION, True)
run_model(CHAINED_REGRESSOR, ['Lag1'], LINEAR_REGRESSION, True)

run_model(MULTI_OUTPUT_REGRESSOR, ['TimeUnit', 'Lag1'], SUPPORT_VECTOR_REGRESSOR, True)
run_model(MULTI_OUTPUT_REGRESSOR, ['TimeUnit'], SUPPORT_VECTOR_REGRESSOR, True)
run_model(MULTI_OUTPUT_REGRESSOR, ['Lag1'], SUPPORT_VECTOR_REGRESSOR, True)
run_model(CHAINED_REGRESSOR, ['TimeUnit', 'Lag1'], SUPPORT_VECTOR_REGRESSOR, True)
run_model(CHAINED_REGRESSOR, ['TimeUnit'], SUPPORT_VECTOR_REGRESSOR, True)
run_model(CHAINED_REGRESSOR, ['Lag1'], SUPPORT_VECTOR_REGRESSOR, True)

# print(stats.to_latex(index=False))
# dump(stats,open('stats.pkl','wb'))


def train_and_tune(model_name: str, save_model: bool = False):
    import tensorflow as tf
    import keras_tuner as kt
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, SimpleRNN, Input
    from keras.losses import MeanSquaredError
    from keras.callbacks import ReduceLROnPlateau
    from keras.callbacks import EarlyStopping
    sc, X_train, y_train, X_test, y_test = ModelUtils.get_data_for_memory_based_model()
    model = Sequential()
    if LONG_SHORT_TERM_MEMORY == model_name:
        model.add(LSTM(60, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    else:
        model.add(SimpleRNN(60, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    model.add(Dense(y_train.shape[1]))
    model.compile(loss=['mse','mse','mse'], optimizer='adam')
    print(model.summary())
    stop_early = EarlyStopping(monitor='loss', patience=5)
    history = model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1, shuffle=False,
                        callbacks=[stop_early])
    predicted = model.predict(X_test)
    run_stats = {'base':{
        'R2_Score': r2_score(y_test, predicted, multioutput='raw_values'),
        'MAE': mae(y_test, predicted, multioutput='raw_values'),
        'MSE': mse(y_test, predicted, multioutput='raw_values')
    }}
    if save_model:
        dump({
            'model': model,
            'scaler': sc,
            'R2_Score': r2_score(y_test, predicted, multioutput='raw_values'),
            'MAE': mae(y_test, predicted, multioutput='raw_values'),
            'MSE': mse(y_test, predicted, multioutput='raw_values')
        }, open('models/'+model_name+'.model', 'wb'))
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        sns.lineplot(arange(len(y_test)), [row[i] for row in y_test], color='blue', label='Actual', ax=ax[i])
        sns.lineplot(arange(len(y_test)), [row[i] for row in predicted], color='red', label='Predicted', ax=ax[i])
        ax[i].set_title(['active_cases', 'cured', 'death'][i])
    plt.show()
    print(run_stats)
    def build(hyperparams):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(
            LSTM(units=hyperparams.Int('units', min_value=16, max_value=512, step=32),
                 activation='relu'))
        model.add(Dense(y_train.shape[1]))
        hp_learning_rate = hyperparams.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=MeanSquaredError(),
                      metrics=['mse'])
        return model
    tuner = kt.Hyperband(build,
                         objective='val_mse',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, shuffle=False, callbacks=[stop_early])
    tuned_model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    tuned_model.summary()
    tuned_model.fit(X_train, y_train, epochs=50, validation_split=0.2, shuffle=False, callbacks=[stop_early])
    predicted = tuned_model.predict(X_test)
    run_stats['tuned']={
        'R2_Score': r2_score(y_test, predicted, multioutput='raw_values'),
        'MAE': mae(y_test, predicted, multioutput='raw_values'),
        'MSE': mse(y_test, predicted, multioutput='raw_values')
    }
    print(run_stats['base']['R2_Score'],"\n", run_stats['tuned']['R2_Score'])
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        sns.lineplot(arange(len(y_test)), [row[i] for row in y_test], color='blue', label='Actual', ax=ax[i])
        sns.lineplot(arange(len(y_test)), [row[i] for row in predicted], color='red', label='Predicted', ax=ax[i])
    plt.show()
    if save_model:
        dump({
            'model': tuned_model,
            'scaler': sc,
            'R2_Score': r2_score(y_test, predicted, multioutput='raw_values'),
            'MAE': mae(y_test, predicted, multioutput='raw_values'),
            'MSE': mse(y_test, predicted, multioutput='raw_values')
        }, open('models/'+model_name+'_tuned.model', 'wb'))
    return run_stats


train_and_tune(RECURRENT_NEURAL_NETWORK, True)
train_and_tune(LONG_SHORT_TERM_MEMORY, True)
