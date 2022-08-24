import dash
from dash import html, dcc, Input, Output, callback
from utils import DataUtils
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from .sidebar import sidebar

dash.register_page(__name__, path='/eda', description="Exploratory Data Analysis", top_nav=True)
state_data = DataUtils.get_state_data()
country_data = DataUtils.get_country_data()


@callback(
    Output('example-graph', 'figure'),
    Input('state', 'value')
)
def update_figure(state):
    if state is None:
        return {}
    fg = px.line(state_data[state_data['state'] == state], x='Date',
                 y=['cured', 'death', 'confirmed', 'confirmed_india', 'confirmed_foreign'], title='Cases Count')
    fg.update_xaxes(rangeslider_visible=True)
    fg.update_layout(transition_duration=500)
    return fg


def get_heatmap():
    import plotly.figure_factory as ff

    corr = country_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask)

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(),
                                      x=df_mask.columns.tolist(),
                                      y=df_mask.columns.tolist(),
                                      colorscale=px.colors.diverging.RdBu,
                                      hoverinfo="none",  # Shows hoverinfo for null values
                                      showscale=True, ygap=1, xgap=1
                                      )

    fig.update_xaxes(side="bottom")

    fig.update_layout(
        title_text='Heatmap',
        title_x=0.5,
        # width=1000,
        # height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed',
        template='plotly_white'
    )

    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    return fig


@callback(
    Output('relation-graph', 'figure'),
    Input('show-heatmap', 'value'),
    Input('relation-x', 'value'),
    Input('relation-y', 'value')
)
def plot_relation(show_heatmap, x_col='cured', y_col='death'):
    show_heatmap_flag = show_heatmap is not None and 'Heatmap' in show_heatmap
    if show_heatmap_flag:
        return get_heatmap()
    if show_heatmap is not None and 'Boxplot' in show_heatmap:
        fig = go.Figure()
        for col in country_data.columns:
            if 'Date' not in col:
                fig.add_trace(go.Box(x=country_data[col], name=col))
        return fig
    if None in [x_col, y_col]:
        return {}
    fg = px.scatter(country_data, x=x_col, y=y_col, trendline='ols')
    return fg


layout = dbc.Row([
                  dbc.Col(html.Div(children=[
                      html.H1('EDA'),
                      dcc.Tabs([
                          dcc.Tab(label='State-wise Analysis', children=[
                              dcc.Dropdown(id='state',
                                           options=[{"label": name, "value": name} for name in
                                                    state_data['state'].unique()]),
                              dcc.Graph(id='example-graph')
                          ]),
                          dcc.Tab(label='Relation', children=[
                              dcc.Checklist(id='show-heatmap', options=['Heatmap', 'Boxplot']),
                              dcc.Dropdown(id='relation-x',
                                           options=[{"label": name, "value": name} for name in country_data.columns]),
                              dcc.Dropdown(id='relation-y',
                                           options=[{"label": name, "value": name} for name in country_data.columns]),
                              dcc.Graph(id='relation-graph')
                          ])
                      ])
                  ]))
                  ])
