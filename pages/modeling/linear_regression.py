import dash
import pandas as pd
from dash import html, dcc, callback, ctx, Input, Output, dash_table
import dash_bootstrap_components as dbc
from utils import DataUtils
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from utils import ModelUtils
from sklearn.metrics import r2_score, mean_absolute_error as mae
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from constants import *

# dash.register_page(__name__, path='/model/linear_regression')
data = DataUtils.get_country_data().copy()
data['TimeUnit'] = np.arange(len(data.index))
data['Lag1'] = data['active_cases'].shift(1).fillna(0)
data['Lag2'] = data['active_cases'].shift(2).fillna(0)
train_size = int(data.shape[0] * 0.75)
train, test = ModelUtils.train_test_split(data, test_size=0.25)
target = ['cured', 'death', 'active_cases']


def plot_actual_vs_predicted(X_test, y_test, predicted, test=None):
    fig = make_subplots(rows=1, cols=3, subplot_titles=y_test.columns)
    col_index = 1
    time_unit_values = X_test['TimeUnit'].values if 'TimeUnit' in X_test else test['TimeUnit'].values
    for index, col in enumerate(y_test.columns):
        mydf = pd.DataFrame(list(zip(time_unit_values, y_test[col].values, [row[index] for row in predicted])),
                            columns=['TimeUnit', 'Actual', 'Predicted'])
        fig.add_trace(go.Scatter(x=mydf['TimeUnit'].values, y=mydf['Actual'].values, name='actual', mode='markers',
                                 legendgroup='actual', marker={'color': 'blue'}, showlegend=col_index == 1),
                      row=1, col=col_index)
        fig.add_trace(
            go.Scatter(x=mydf['TimeUnit'].values, y=mydf['Predicted'].values, name='predicted', mode='markers',
                       legendgroup='predicted', marker={'color': 'red'}, showlegend=col_index == 1),
            row=1, col=col_index)
        col_index += 1
    return fig


# @callback(
#     Output('result', 'children'),
#     Output('perf', 'figure'),
#     Input('features', 'value'),
#     Input('lr-model-choice', 'value'),
#     Input('base_model_choice', 'value'),
#     Input('lr_train_test', 'n_clicks')
# )
def button_clicked(features, model_choice, base_model_choice, btn):
    output = {}
    if ctx.triggered_id is None or features is None or len(features) == 0:
        output = {}
    if ctx.triggered_id == 'lr_train_test':
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]
        base_model = LinearRegression() if base_model_choice == LINEAR_REGRESSION else SVR(kernel='poly')
        model = MultiOutputRegressor(base_model) if MULTI_OUTPUT_REGRESSOR == model_choice \
            else RegressorChain(base_model)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        cols = ['Metric']
        cols.extend(target)
        stats = pd.DataFrame(columns=cols)
        stats.set_index('Metric', inplace=True)
        stats.loc['R2_score'] = r2_score(y_test, predicted, multioutput='raw_values')
        stats.loc['MAE'] = mae(y_test, predicted, multioutput='raw_values')
        eqns = []
        if base_model_choice == SUPPORT_VECTOR_REGRESSOR:
            print(model.estimators_[0].support_vectors_)
        try:
            for index, target_variable in enumerate(target):
                eqn = ''
                if 'Chained' in model_choice:
                    print(features," ",model.estimators_[index].coef_)
                for col, coef in zip(features, model.estimators_[index].coef_):
                    if coef < 0:
                        eqn = eqn[:-1]
                    eqn += str(round(coef, 2)) + "*" + col + " +"
                eqn = eqn[:-1] + str(round(model.estimators_[0].intercept_))
                eqns.append(eqn)
            stats.loc['Equation'] = eqns
        except Exception as e:
            print(e)
        stats.reset_index(inplace=True)
        output = str(r2_score(y_test, predicted, multioutput='raw_values'))
        return html.Div([
            html.H1('Regression Equation is ' + eqn),
            html.H2('R2 value is' + str(output)),
            dash_table.DataTable(stats.to_dict('records'), [{'name': i, 'id': i} for i in stats.columns])
        ]), plot_actual_vs_predicted(X_test, y_test, predicted, test)
    return output, {}


layout = html.Div(children=[html.H1('Linear Regression'),
                            dcc.RadioItems(id='lr-model-choice',
                                           options=[MULTI_OUTPUT_REGRESSOR, CHAINED_REGRESSOR],
                                           value=MULTI_OUTPUT_REGRESSOR),
                            dcc.RadioItems(id='base_model_choice',
                                           options=[LINEAR_REGRESSION, SUPPORT_VECTOR_REGRESSOR],
                                           value=LINEAR_REGRESSION),
                            dcc.Checklist(id='features', options=['TimeUnit', 'Lag1']),
                            html.Button('Train And Test', id='lr_train_test'),
                            html.Div(id='result'),
                            dcc.Graph('perf')
                            ])
