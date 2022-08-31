import dash
import numpy as np
import pandas as pd
from dash import html, dcc, dash_table
from utils import DataUtils
import dash_bootstrap_components as dbc
import plotly.express as px

dash.register_page(__name__, path='/', top_nav=True)


def get_top_5(data_to_process: pd.DataFrame, old_data: pd.DataFrame, by: str, ascending: bool = True):
    top5_data = data_to_process.sort_values(by, ascending=ascending).head(5)
    day_changes_values = []
    for state in top5_data['state']:
        prev_value = old_data[old_data['state'] == state][by].values
        current_value = data_to_process[data_to_process['state'] == state][by].values
        percent_change = (prev_value - current_value) / current_value * 100
        day_changes_values.append(round(percent_change[0], 2))
    top5_data['day_changes(in %)'] = day_changes_values
    top5_data['% of total'] = round(top5_data[by] / sum(data_to_process[by]) * 100)
    return top5_data[['state', '% of total', 'day_changes(in %)']]


country_data = DataUtils.get_country_data()
state_data = DataUtils.get_state_data(False)
latest_date = '2022-08-29'
before_latest_date = '2022-08-27'
latest_country_data = country_data[country_data['Date'] == latest_date]
before_latest_data = state_data[state_data['Date'] == before_latest_date]
latest_state_data = state_data[state_data['Date'] == latest_date]


def show_trend(value):
    img_url = 'https://www.freeiconspng.com/thumbs/up-arrow-png/up-arrow-png-24.png'
    color = 'green'
    if float(value) < 0:
        img_url = 'https://www.freepnglogos.com/uploads/arrow-png/file-arrow-down-2.png'
        color = 'red'
    return html.Td([str(value) + '%', html.Img(src=img_url, width=20)], style={'color': color})


def prepare_table(data_to_convert, col_transformation_map={}):
    table_headers = []
    columns = data_to_convert.columns
    for col in columns:
        table_headers.append(html.Th(col))
    table_head = html.Thead(html.Tr(table_headers))
    rows = []
    for data_row in data_to_convert.values:
        row = []
        for index, val in enumerate(data_row):
            if columns[index] in col_transformation_map:
                row.append(col_transformation_map[columns[index]](val))
            else:
                row.append(html.Td(val))
        rows.append(html.Tr(row))
    table_body = html.Tbody(rows)
    return dbc.Table([table_head] + [table_body])


cards = []
for col in ['confirmed', 'cured', 'death']:
    cards.append(dbc.Card([dbc.CardHeader('As of ' + latest_date),
                           dbc.CardBody([
                               html.H4(col, className='card-title'),
                               html.H5(sum(latest_state_data[col]), className='card-subtitle'),
                               prepare_table(get_top_5(latest_state_data, before_latest_data, col, False),
                                             {'day_changes(in %)': show_trend})
                           ])]))

cards.insert(0, dbc.Card([dbc.CardHeader("Nation level Statistics"),
                          dbc.CardBody([
                              prepare_table(latest_country_data.drop(columns=['Date'])),
                              html.H4('Percentage of People cured: ' + str(round((latest_country_data['cured'].values /
                                                                                  latest_country_data[
                                                                                      'total_confirmed_cases'].values * 100)[
                                                                                     0], 2)) + '%'),
                              html.H4('Percentage of People dead: ' + str(round((latest_country_data['death'].values /
                                                                                 latest_country_data[
                                                                                     'total_confirmed_cases'].values * 100)[
                                                                                    0], 2)) + '%')

                          ])]))
row = []
for card in cards:
    row.append(dbc.Col(card))


def get_plot():
    country_data['cured'] = country_data['cured'].diff().fillna(0)
    country_data['confirmed_cases'] = country_data['total_confirmed_cases'].diff().fillna(0)
    fg = px.line(country_data, x='Date', y=['active_cases', 'cured', 'death', 'confirmed_cases'], title='Nation-wide details')
    fg.update_xaxes(rangeslider_visible=True)
    fg.update_layout(transition_duration=500)
    return fg


layout = html.Div([dbc.Row(dcc.Graph('country-level', figure=get_plot())),
                   dbc.Row(row)])
