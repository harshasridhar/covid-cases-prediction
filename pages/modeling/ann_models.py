from utils import ModelUtils
import dash
from dash import html, dcc, callback, Input, Output
from constants import *
from plotly.subplots import make_subplots
from pandas import DataFrame
from numpy import arange
from plotly.graph_objects import Scatter
from time import time

dash.register_page(__name__, path='/model/memory')
n_clicks = 0

def log(message, extra={}):
    print(str({'time': time(), 'message': message, 'extra': extra}))

@callback(
    Output('mem_model_stats', 'figure'),
    Input('memory_model_choice', 'value'),
    Input('tuned_model', 'value'),
    Input('mem_test_model', 'n_clicks')
)
def display_stats(memory_model_choice, tuned_model, mem_test_model):
    global n_clicks
    log('Here')
    if mem_test_model and mem_test_model > n_clicks:
        n_clicks = mem_test_model
        saved_model = ModelUtils.get_pickled_model(memory_model_choice,features=None, get_tuned_model=not tuned_model)
        log('Model Retrieved')
        model = saved_model['model']
        saved_model = None
        sc, X_train, y_train, X_test, y_test = ModelUtils.get_data_for_memory_based_model()
        log('Retrieved Data')
        sc, X_train, y_train = None, None, None
        predictions = model.predict(X_test)
        log('Made Predictions')
        target_variables = ['active_cases', 'cured', 'death']
        fig = make_subplots(rows=1, cols=3, subplot_titles=target_variables)
        col_index = 1
        for index, col in enumerate(target_variables):
            df = DataFrame(list(zip(arange(len(y_test)),[row[index] for row in y_test], [row[index] for row in predictions])),
                           columns=['TimeUnit','Actual', 'Predicted'])
            fig.add_trace(Scatter(x=df['TimeUnit'].values, y=df['Actual'].values, name='actual', mode='markers',
                                  legendgroup='actual', marker={'color': 'blue'}, showlegend=col_index == 1), row=1,
                          col=col_index)
            fig.add_trace(Scatter(x=df['TimeUnit'].values, y=df['Predicted'].values, name='actual', mode='markers',
                                  legendgroup='predicted', marker={'color': 'red'}, showlegend=col_index == 1), row=1,
                          col=col_index)
            col_index += 1
        return fig
    return {}


layout = html.Div(children=[
    dcc.RadioItems(id='memory_model_choice',
                   options=[RECURRENT_NEURAL_NETWORK, LONG_SHORT_TERM_MEMORY],
                   value=RECURRENT_NEURAL_NETWORK),
    dcc.Checklist(id='tuned_model', options=['Tuned Model']),
    html.Button('Test Model', id='mem_test_model'),
    dcc.Graph('mem_model_stats')
])