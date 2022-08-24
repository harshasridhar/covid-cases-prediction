import dash
from dash import html
from .sidebar import sidebar
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', top_nav=True)
layout = dbc.Row([
                  dbc.Col(html.H1('Home Page'))
                  ])
