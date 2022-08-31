from scipy.integrate import odeint
from numpy import linspace, arange
from plotly.graph_objects import Scatter
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__, '/models/epidemiological_models')

population = 1380004385
R0 = 1.65


def compute_derivative_for_sir(y, t, beta, gamma):
    s, i, r = y
    ds_dt = -beta * s * i
    di_dt = (beta * s * i) - (gamma * i)
    dr_dt = gamma * i
    return [ds_dt, di_dt, dr_dt]


def simulate_sir_model(num_days):
    t_infective = 8.4
    gamma = 1 / t_infective
    beta = R0 * gamma
    initial_infected_proportion = 1 / population
    initial_recovered_proportion = 0.0
    initial_suscepted_proportion = 1 - initial_infected_proportion - initial_recovered_proportion
    y_initial = initial_suscepted_proportion, initial_infected_proportion, initial_recovered_proportion
    time_steps = linspace(0, num_days, num_days - 50)
    solution = odeint(compute_derivative_for_sir, y_initial, time_steps, args=(beta, gamma))
    return solution.T


def simulate_seir_model(num_days):
    def derivate(y, t, alpha, beta, gamma):
        s, e, i, r = y
        ds_dt = -beta * s * i
        de_dt = (beta * s * i) - (alpha * e)
        di_dt = (alpha * e) - (gamma * i)
        dr_dt = gamma * i
        return [ds_dt, de_dt, di_dt, dr_dt]

    incubation_time = 5.1
    infection_time = 3.3
    initial_exposed_proportion = 1 / population
    initial_infected_proportion = 0.0
    initial_recovered_proportion = 0.0
    initial_suscepted_proportion = 1 - initial_exposed_proportion - initial_infected_proportion - initial_recovered_proportion
    alpha = 1 / incubation_time
    gamma = 1 / infection_time
    beta = R0 * gamma
    y_inital = initial_suscepted_proportion, initial_exposed_proportion, initial_infected_proportion, initial_recovered_proportion
    time_steps = linspace(0, num_days, num_days - 50)
    solution = odeint(derivate, y_inital, time_steps, args=(alpha, beta, gamma))
    return solution.T


@callback(
    Output('epidemiological_models', 'figure'),
    Input('days', 'value')
)
def get_simulation_plots(days):
    if days is None or len(days) == 0:
        return {}
    s, i, r = simulate_sir_model(int(days))
    fig = make_subplots(1, 2, x_title='Time/Days', y_title='Proportion of Population',
                        subplot_titles=['SIR Model', 'SEIR Model'])
    fig.add_trace(Scatter(x=arange(len(s)), y=s, name='Susceptible', marker={'color': 'blue'}), row=1, col=1)
    fig.add_trace(Scatter(x=arange(len(s)), y=i, name='Infected', marker={'color': 'red'}), row=1, col=1)
    fig.add_trace(Scatter(x=arange(len(s)), y=r, name='Susceptible', marker={'color': 'green'}), row=1, col=1)

    s, e, i, r = simulate_seir_model(int(days))
    fig.add_trace(Scatter(x=arange(len(s)), y=s, name='Susceptible', marker={'color': 'blue'}), row=1, col=2)
    fig.add_trace(Scatter(x=arange(len(s)), y=e, name='Exposed', marker={'color': 'orange'}), row=1, col=2)
    fig.add_trace(Scatter(x=arange(len(s)), y=i, name='Infected', marker={'color': 'red'}), row=1, col=2)
    fig.add_trace(Scatter(x=arange(len(s)), y=r, name='Susceptible', marker={'color': 'green'}), row=1, col=2)
    return fig


layout = html.Div([
    html.H2("Epidemiologial Models"),
    dcc.Dropdown(id='days', options=['500', '750', '1000', '1500'], placeholder='Select Days to Simulate'),
    dcc.Graph('epidemiological_models')
])
