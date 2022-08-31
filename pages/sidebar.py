import dash
from dash import html
import dash_bootstrap_components as dbc


def sidebar():
    # This doesn't work as the order of the page registration is different,
    # leading to diff number of elements in the sidebar accordingly
    # return html.Div(
    #     dbc.Nav([
    #         dbc.NavLink(page['name'], href=page['path'], active='exact')
    #         for page in dash.page_registry.values()
    #     ], vertical=True, pills=True, className='bg-light')
    # )
    # return html.Div(
    #     dbc.Nav([
    #         dbc.NavLink('Home', href='/', active='exact'),
    #         dbc.NavLink('EDA', href='/eda', active='exact'),
    #         dbc.DropdownMenu("Modeling", children=[
    #             dbc.DropdownMenuItem("LR", href="/model/linear_regression")
    #         ], nav=True, in_navbar=True)
    #     ], vertical=True, pills=True, className='bg-light')
    # )
    return dbc.NavbarSimple(children=[
        dbc.NavItem(dbc.NavLink('Home', href='/', active='exact')),
        dbc.NavItem(dbc.NavLink('EDA', href='/eda', active='exact')),
        dbc.DropdownMenu(children=[
            dbc.DropdownMenuItem("LR", href="/model/linear_regression"),
            dbc.DropdownMenuItem("Memory", href="/model/memory"),
            dbc.DropdownMenuItem("TimeSeries", href="/model/time_series"),
            dbc.DropdownMenuItem("Epidemiological Models", href="/models/epidemiological_models")
        ], nav=True, in_navbar=True, label="Modeling")
    ])

