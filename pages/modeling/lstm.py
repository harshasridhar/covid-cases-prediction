import matplotlib.pyplot as plt
import dash
from sklearn.preprocessing import MinMaxScaler
from utils import DataUtils, ModelUtils
from numpy import reshape, arange
from dash import html, dcc, callback, Input, Output, ctx
from constants import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# dash.register_page(__name__, path='/model/memory')
sc = MinMaxScaler(feature_range=(0, 1))
data = DataUtils.get_country_data()


def plot_actual_vs_predicted(predicted, y_test):
    predicted = reshape(sc.inverse_transform(predicted), len(predicted))
    y_test = reshape(sc.inverse_transform(y_test.reshape(-1, 1)), len(y_test))
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=arange(len(y_test)), y=y_test, name='actual', mode='markers', marker={'color': 'blue'}),
                  row=1, col=1)
    fig.add_trace(
        go.Scatter(x=arange(len(y_test)), y=predicted, name='actual', mode='markers', marker={'color': 'blue'}),
        row=1, col=1)
    return fig


# @callback(
#     Output('predictions', 'figure'),
#     Input('model_choice', 'value'),
#     Input('feature', 'value')
# )
def train_test(model_choice, feature):
    if ctx.triggered_id == 'lr_train_test':
        from keras.callbacks import ReduceLROnPlateau
        from keras.utils.vis_utils import plot_model
        scaled_data = sc.fit_transform(data[feature].values.reshape(-1, 1))
        time_steps = 15
        X, y = ModelUtils.get_XY_from_data(scaled_data, time_steps)
        X_train, X_test = ModelUtils.train_test_split(X, test_size=0.25)
        y_train, y_test = ModelUtils.train_test_split(y, test_size=0.25)
        X_train = reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        num_nodes = 60
        model = ModelUtils.get_model(model_choice, num_nodes)

        model.compile(loss=MEAN_SQUARED_ERROR,
                      optimizer=ADAM_OPTIMIZER,
                      metrics=[MEAN_SQUARED_ERROR, ModelUtils.r_square])
        print(model.summary())
        plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
        batchsize = 100
        epochs = 100
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_mean_squared_error',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=1e-10)
        history = model.fit(X_train,
                            y_train,
                            batch_size=batchsize,
                            epochs=epochs,
                            validation_split=0.2,
                            shuffle=False,
                            callbacks=[learning_rate_reduction])

        predicted = model.predict(X_test)
        return plot_actual_vs_predicted(predicted, y_test)
    else:
        {}


layout = html.Div(children=[
    dcc.RadioItems(id='model_choice',
                   options=[RECURRENT_NEURAL_NETWORK, LONG_SHORT_TERM_MEMORY],
                   value=RECURRENT_NEURAL_NETWORK),
    dcc.RadioItems(id='feature',
                   options=data.columns),
    html.Button('Train and Test', id='train_test'),
    dcc.Graph('predictions')
])
