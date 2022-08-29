import dash
from dash import html, dcc, Input, Output, callback, ctx
from constants import *
from utils import ModelUtils
from plotly.subplots import make_subplots
from pandas import DataFrame, Series
from numpy import arange
from plotly.graph_objects import Scatter


def get_options_from_list(list_elements):
    return [{'label': element, 'value': element} for element in list_elements]


dash.register_page(__name__, path='/model/time_series')


@callback(
    Output('time_series_outputs', 'figure'),
    Output('ar_based_model_outputs', 'figure'),
    Input('target_var', 'value'),
    Input('time_series_test', 'n_clicks')
)
def handle_change(target_var, time_series_test):
    if ctx.triggered_id == 'time_series_test':
        train, test = ModelUtils.get_data_for_time_series_models(target_var)
        ses_model = ModelUtils.get_pickled_model(SIMPLE_EXPONENTIAL_SMOOTHING, [target_var])['model']
        des_model = ModelUtils.get_pickled_model(DOUBLE_EXPONENTIAL_SMOOTHING, [target_var])['model']
        tes_model = ModelUtils.get_pickled_model(TRIPLE_EXPONENTIAL_SMOOTHING, [target_var])['model']
        fig = make_subplots(1, 3, subplot_titles=[SIMPLE_EXPONENTIAL_SMOOTHING,
                                                  DOUBLE_EXPONENTIAL_SMOOTHING,
                                                  TRIPLE_EXPONENTIAL_SMOOTHING])
        col_index = 1
        for index, model in enumerate([ses_model, des_model, tes_model]):
            predictions = Series(model.forecast(len(test))).reset_index(drop=True)
            df = DataFrame(list(zip(arange(len(test)), test, predictions)),
                           columns=['TimeUnit', 'Actual', 'Predicted'])
            fig.add_trace(Scatter(x=df['TimeUnit'].values, y=df['Actual'].values, name='actual', mode='markers',
                                  legendgroup='actual', marker={'color': 'blue'}, showlegend=col_index == 1), row=1,
                          col=col_index)
            fig.add_trace(Scatter(x=df['TimeUnit'].values, y=df['Predicted'].values, name='predicted', mode='markers',
                                  legendgroup='predicted', marker={'color': 'red'}, showlegend=col_index == 1), row=1,
                          col=col_index)
            col_index += 1
        arima_model = ModelUtils.get_pickled_model(ARIMA, [target_var])['model']
        sarima_model = ModelUtils.get_pickled_model(SARIMA, [target_var])['model']
        col_index = 1
        fig2 = make_subplots(1, 2, subplot_titles=[ARIMA, SARIMA])
        for index, model in enumerate([arima_model, sarima_model]):
            predictions = Series(model.predict(len(test))).reset_index(drop=True)
            df = DataFrame(list(zip(arange(len(test)), test, predictions)),
                           columns=['TimeUnit', 'Actual', 'Predicted'])
            fig2.add_trace(Scatter(x=df['TimeUnit'].values, y=df['Actual'].values, name='actual', mode='markers',
                                  legendgroup='actual', marker={'color': 'blue'}, showlegend=col_index == 1), row=1,
                          col=col_index)
            fig2.add_trace(Scatter(x=df['TimeUnit'].values, y=df['Predicted'].values, name='predicted', mode='markers',
                                  legendgroup='predicted', marker={'color': 'red'}, showlegend=col_index == 1), row=1,
                          col=col_index)
            col_index += 1
        return fig, fig2
    return {}, {}


layout = html.Div(children=[
    dcc.Dropdown(id='target_var',
                 options=get_options_from_list(TARGET_COLUMNS)),
    html.Button('Test Models', id='time_series_test'),
    dcc.Graph('time_series_outputs'),
    dcc.Graph('ar_based_model_outputs')
])
