import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
from pages.sidebar import sidebar

app = Dash(__name__,
           use_pages=True,
           external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css"])

app.layout = html.Div([
    html.H1('Predicting the number of COVID cases in India'),
    dbc.Row([sidebar()]),
    dbc.Row([dash.page_container])
])

if __name__ == '__main__':
    app.run_server(debug=True)
