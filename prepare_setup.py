import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from pandas import DataFrame, Series
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from pmdarima import auto_arima

data = DataUtils.get_country_data().copy()
data['TimeUnit'] = arange(data.shape[0])
stats = DataFrame(columns=['Model', 'Base', 'Features', 'R2_Score', 'MAE', 'MSE'])
index = 0
multi_output_target = ['active_cases', 'cured', 'death']


def get_metrics(y_true, predicted, is_multioutput: bool = False):
    R2 = r2_score(y_true, predicted, multioutput='raw_values') if is_multioutput else r2_score(y_true, predicted)
    MAE = mae(y_true, predicted, multioutput='raw_values') if is_multioutput else mae(y_true, predicted)
    MSE = mse(y_true, predicted, multioutput='raw_values') if is_multioutput else mse(y_true, predicted)
    return {
        'R2_score': R2,
        'MAE': MAE,
        'MSE': MSE
    }


def run_model(model_name: str, features: list, base_model: str = None, save_model: bool = False):
    filename = ''
    orignal_feature_list = features.copy()
    if base_model is None:
        filename = 'models/{}_[{}].model'.format(model_name, ','.join(features))
    else:
        filename = 'models/{}_{}_[{}].model'.format(model_name, base_model, ','.join(features))
    model = None
    if model_name in [MULTI_OUTPUT_REGRESSOR, CHAINED_REGRESSOR]:
        base_model_obj = LinearRegression() if base_model == LINEAR_REGRESSION else SVR(kernel='poly')
        model = MultiOutputRegressor(base_model_obj) if model_name == MULTI_OUTPUT_REGRESSOR else RegressorChain(
            base_model_obj)
    target_columns = ['cured', 'death', 'active_cases']
    X_train, y_train, X_test, y_test = ModelUtils.get_data_for_linear_model(features, target_columns)
    start_time = time()
    model.fit(X_train, y_train)
    time_taken = time() - start_time
    predictions = model.predict(X_test)
    eqns=[]
    if isinstance(model, MultiOutputRegressor) and isinstance(base_model_obj, LinearRegression):
        for ind, col in enumerate(target_columns):
            eqn = col + ' = '
            for coef, col in zip(model.estimators_[ind].coef_, X_train.columns):
                eqn += '{}{} * {} '.format('+' if coef > 0 else '', round(coef, 4), col)
            eqns.append(eqn)
    r2 = list(map(partial(round, ndigits=3), r2_score(y_test, predictions, multioutput='raw_values')))
    MAE = list(map(partial(round, ndigits=3), mae(y_test, predictions, multioutput='raw_values')))
    MSE = list(map(partial(round, ndigits=3), mse(y_test, predictions, multioutput='raw_values')))
    # plt.figure()
    # sns.lineplot(arange(len(y_test)), y_test['active_cases'].values, color='blue', label='Actual')
    # sns.lineplot(arange(len(y_test)), [row[2] for row in predictions], color='red', label='Predicted')
    # plt.show()
    global index
    stats.loc[index] = [model_name, base_model, ','.join(orignal_feature_list), r2, MAE, MSE]
    index += 1
    print(filename)
    print(r2)
    if save_model:
        dump({'model': model,
              # 'equations': eqns,
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

print(stats.to_latex(index=False))
dump(stats,open('stats.pkl','wb'))

memory_model_stats = pd.DataFrame(columns=['Model', 'R2_Score', ' MAE', 'MSE'])
memory_model_stats.set_index('Model', inplace=True)

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
    model.compile(loss=['mse', 'mse', 'mse'], optimizer='adam')
    print(model.summary())
    stop_early = EarlyStopping(monitor='loss', patience=5)
    history = model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=1, shuffle=False,
                        callbacks=[stop_early])
    predicted = model.predict(X_test)
    run_stats = {'base': {
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
        }, open('models/' + model_name + '.model', 'wb'))
    # fig, ax = plt.subplots(1, 3)
    # for i in range(3):
    #     sns.lineplot(arange(len(y_test)), [row[i] for row in y_test], color='blue', label='Actual', ax=ax[i])
    #     sns.lineplot(arange(len(y_test)), [row[i] for row in predicted], color='red', label='Predicted', ax=ax[i])
    #     ax[i].set_title(['active_cases', 'cured', 'death'][i])
    # plt.show()
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
    run_stats['tuned'] = {
        'R2_Score': r2_score(y_test, predicted, multioutput='raw_values'),
        'MAE': mae(y_test, predicted, multioutput='raw_values'),
        'MSE': mse(y_test, predicted, multioutput='raw_values')
    }
    r2 = list(map(partial(round, ndigits=3), r2_score(y_test, predicted, multioutput='raw_values')))
    MAE = list(map(partial(round, ndigits=3), mae(y_test, predicted, multioutput='raw_values')))
    MSE = list(map(partial(round, ndigits=3), mse(y_test, predicted, multioutput='raw_values')))

    memory_model_stats.loc[model_name] = [r2, MAE, MSE]
    print(run_stats['base']['R2_Score'], "\n", run_stats['tuned']['R2_Score'])
    # fig, ax = plt.subplots(1, 3)
    # for i in range(3):
    #     sns.lineplot(arange(len(y_test)), [row[i] for row in y_test], color='blue', label='Actual', ax=ax[i])
    #     sns.lineplot(arange(len(y_test)), [row[i] for row in predicted], color='red', label='Predicted', ax=ax[i])
    # plt.show()
    if save_model:
        dump({
            'model': tuned_model,
            'scaler': sc,
            'R2_Score': r2_score(y_test, predicted, multioutput='raw_values'),
            'MAE': mae(y_test, predicted, multioutput='raw_values'),
            'MSE': mse(y_test, predicted, multioutput='raw_values')
        }, open('models/' + model_name + '_tuned.model', 'wb'))
    return run_stats


train_and_tune(RECURRENT_NEURAL_NETWORK, True)
train_and_tune(LONG_SHORT_TERM_MEMORY, True)
print(memory_model_stats.to_latex())
dump(memory_model_stats,open('memory_model_stats.pkl','wb'))


def run_time_series_models(model_name: str, target_column: str, trend: str = 'mul', seasonal: str = 'mul', seasonal_periods: int = 10, save_model: bool = False):
    train, test = ModelUtils.get_data_for_time_series_models(target_column)
    print(model_name, ' ', target_column)
    model = None
    input_seasonality = 12
    if ARIMA == model_name:
        model = auto_arima(train, exogenous=None,
                           start_p=1, start_q=1,
                           max_p=3, max_q=3, m=input_seasonality,
                           start_P=0, seasonal=False,
                           d=None, max_D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True, stepwise=True,
                           max_order=12)
    elif SARIMA == model_name:
        model = auto_arima(train, exogenous=None,
                           start_p=1, start_q=1,
                           max_p=3, max_q=3, m=input_seasonality,
                           start_P=0, seasonal=True,
                           d=None, max_D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True, stepwise=True,
                           max_order=12)
    if SIMPLE_EXPONENTIAL_SMOOTHING == model_name:
        model = SimpleExpSmoothing(np.asarray(train)).fit(optimized=True)
    elif DOUBLE_EXPONENTIAL_SMOOTHING == model_name:
        model = ExponentialSmoothing(np.asarray(train), trend=trend).fit(optimized=True)
    elif TRIPLE_EXPONENTIAL_SMOOTHING == model_name:
        model = ExponentialSmoothing(np.asarray(train), trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit(optimized=True)
    predicted = None
    if model_name in [ARIMA, SARIMA]:
        predicted = Series(model.predict(len(test))).reset_index(drop=True)
    else:
        predicted = Series(model.forecast(len(test))).reset_index(drop=True)
    run_stats = get_metrics(test, predicted)
    print(run_stats)
    if save_model:
        run_stats['model'] = model
        filename = 'models/{}_{}.model'.format(model_name, target_column)
        dump(run_stats, open(filename, 'wb'))

success=[]

for model in [SIMPLE_EXPONENTIAL_SMOOTHING, DOUBLE_EXPONENTIAL_SMOOTHING, TRIPLE_EXPONENTIAL_SMOOTHING,ARIMA, SARIMA]:
    for col in multi_output_target:
        try:
            run_time_series_models(model, target_column=col, save_model=True)
            success.append([model, col])
        except Exception as e:
            print('Exception OCCURED',str(e))

run_time_series_models(DOUBLE_EXPONENTIAL_SMOOTHING, target_column='cured', trend='add', save_model=True)
run_time_series_models(DOUBLE_EXPONENTIAL_SMOOTHING, target_column='death', trend='add', save_model=True)
run_time_series_models(TRIPLE_EXPONENTIAL_SMOOTHING, target_column='cured', save_model=True,trend='add', seasonal='add')
run_time_series_models(TRIPLE_EXPONENTIAL_SMOOTHING, target_column='death', save_model=True,trend='add', seasonal='add')
print(success)
