from utils import ModelUtils
import dash
from dash import html, dcc, callback, Input, Output, ctx
from constants import *
from plotly.graph_objects import Scatter
from plotly.subplots import make_subplots
from numpy import arange
from pandas import DataFrame


dash.register_page(__name__, path='/model/linear_regression')


def get_actual_vs_predicted_plot(actual: DataFrame, predicted):
    fig = make_subplots(rows=1, cols=3, subplot_titles=actual.columns)
    index_values = arange(len(actual))
    col_index = 1
    for index, col in enumerate(actual.columns):
        df = DataFrame(list(zip(index_values, actual[col].values, [row[index] for row in predicted])),
                       columns=['TimeUnit', 'Actual', 'Predicted'])
        fig.add_trace(Scatter(x=df['TimeUnit'].values, y=df['Actual'].values, name='actual', mode='markers',
                              legendgroup='actual', marker={'color': 'blue'}, showlegend=col_index == 1), row=1,
                      col=col_index)
        fig.add_trace(Scatter(x=df['TimeUnit'].values, y=df['Predicted'].values, name='predicted', mode='markers',
                              legendgroup='predicted', marker={'color': 'red'}, showlegend=col_index == 1), row=1,
                      col=col_index)
        col_index += 1
    return fig


@callback(
    Output('test_stats', 'figure'),
    Input('model_choice', 'value'),
    Input('base_model_choice', 'value'),
    Input('features', 'value'),
    Input('test_model', 'n_clicks')
)
def test_model(model_choice, base_model_choice, features, test_model):
    if ctx.triggered_id == 'test_model':
        features = sorted(features, reverse=True)
        saved_model = ModelUtils.get_pickled_model(model_choice, features, base_model_choice)
        model = saved_model['model']
        X_train, y_train, X_test, y_test = ModelUtils.get_data_for_linear_model(features, ['active_cases','cured','death'])
        predictions = model.predict(X_test)
        return get_actual_vs_predicted_plot(y_test, predictions)
    return {}


layout = html.Div(children=[
    dcc.RadioItems(id='model_choice',
                   options=[MULTI_OUTPUT_REGRESSOR, CHAINED_REGRESSOR],
                   value=MULTI_OUTPUT_REGRESSOR),
    dcc.RadioItems(id='base_model_choice',
                   options=[LINEAR_REGRESSION, SUPPORT_VECTOR_REGRESSOR],
                   value=LINEAR_REGRESSION),
    dcc.Checklist(id='features',
                  options=['TimeUnit', 'Lag1']),
    html.Button('Test Model', id='test_model'),
    dcc.Graph('test_stats')
])
