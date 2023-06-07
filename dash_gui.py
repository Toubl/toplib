import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
from dash_extensions import Keyboard
from dash.dependencies import Input, Output, State
import webbrowser
import platform
import subprocess

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])


card_create_json = dbc.Card([
    dbc.CardHeader('Define the topology optimization problem'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.P('nelx')
            ]),
            dbc.Col([
                dbc.Input(value=200, id='nelx_input')
            ])
        ], style={'marginTop':3}),
        dbc.Row([
            dbc.Col([
                html.P('nely')
            ]),
            dbc.Col([
                dbc.Input(value=100, id='nely_input')
            ])
        ], style={'marginTop':3}),
        dbc.Row([
            dbc.Col([
                html.P('volfrac')
            ]),
            dbc.Col([
                dbc.Input(value=0.35, id='volfrac_input')
            ])
        ], style={'marginTop':3}),
        dbc.Row([
            dbc.Col([
                html.P('penalty factor')
            ]),
            dbc.Col([
                dbc.Input(value=4, id='penalty_input')
            ])
        ], style={'marginTop':3}),
        dbc.Row([
            dbc.Col([
                html.P('filter rmin')
            ]),
            dbc.Col([
                dbc.Input(value=2.4, id='rmin_input')
            ])
        ], style={'marginTop':3}),
        
    ]),
    dbc.CardFooter([
        dbc.Row([
            dbc.Col(width=1),
            dbc.Col([
                dbc.Button('Preview', id='preview_configuration_button', color='dark')
            ], width='auto'),
            dbc.Col([
                dbc.Button('Create json file', id='create_json_button', color='dark')
            ], width='auto')
        ], justify='end', style={'marginTop': '0px'})
    ])
], style={"width": "100%", "margin": "auto", "marginTop": "1rem"})

card_run_problem = dbc.Card([
    dbc.CardHeader('Select and run problem'),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Input(placeholder = 'Choose path...')
            ])
        ], style={'marginTop':3}),
        dbc.Row([
            dbc.Col([
                    dbc.Button("Open Explorer", id="open-explorer-button", color="dark")
            ])
        ], style={'marginTop':3}),
    ]),
    dbc.CardFooter([

    ])
],style={"width": "100%", "margin": "auto", "marginTop": "1rem"})


navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row([
                dbc.Col([
                    html.H4('Topopt at TUM LPL'),
                ]),
                dbc.Col([
                    dbc.Button('Define Problem', color = 'dark'),
                ], width=3),
                dbc.Col([
                    dbc.Button('View Results', color = 'dark'),
                ], width=3),
            ]),
        ]
    ),
    color="light",
    dark=True,
    # className="mb-5",
)


layout = dbc.Container(
    [
        dbc.Row([
            navbar
        ]),
        dbc.Row(
            [
                dbc.Col([
                    card_create_json,
                    card_run_problem,
                ],
                    md=3,
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id='plot_area',
                                config={'displayModeBar': False},
                                responsive=True  # Enable responsiveness to fill available space
                            ),
                        ])
                    ], style={"width": "100%", "margin": "auto", "marginTop": "1rem"}),
                    md=9,
                ),
                Keyboard(id="keyboard"),
                dbc.Button('Log Workout', id='save_button', n_clicks=0, style={'display': 'none'}),
            ]
        ),
    ],
    fluid=True,
)

def open_file_explorer():
    system = platform.system()
    if system == "Windows":
        subprocess.Popen('explorer /select,"."')
    elif system == "Darwin":
        subprocess.Popen(["open", "-R", "."])
    elif system == "Linux":
        subprocess.Popen(["xdg-open", "."])

@app.callback(Output("open-explorer-button", "n_clicks"), [Input("open-explorer-button", "n_clicks")])
def open_explorer(n_clicks):
    if n_clicks:
        open_file_explorer()
    return None

@app.callback(
    Output('plot_area', 'figure'),
    [Input('preview_configuration_button', 'n_clicks')],
    [State('nelx_input', 'value'), State('nely_input', 'value')]
)
def update_graph(n_clicks, nelx_value, nely_value):
    if n_clicks is None:
        return go.Figure()

    nelx_value = float(nelx_value)
    nely_value = float(nely_value)

    # Define your rectangle coordinates
    x0, y0 = 0, 0  # coordinates for the lower left point
    x1, y1 = nelx_value, nely_value  # coordinates for the upper right point

    rectangle = go.layout.Shape(
    type="rect", 
    x0=x0, 
    y0=y0, 
    x1=x1, 
    y1=y1,
    line=dict(color="rgb(0,0,0)", width=2),  # Use RGB color code for black
    fillcolor="RoyalBlue", 
    opacity=0.3,
    )





    # Create a grid with plotly.graph_objs
    grid = []
    for i in range(1, int(nelx_value)):
        grid.append(go.layout.Shape(type="line", x0=i, y0=0, x1=i, y1=nely_value, line=dict(color="Black", width=1)))
    for i in range(1, int(nely_value)):
        grid.append(go.layout.Shape(type="line", x0=0, y0=i, x1=nelx_value, y1=i, line=dict(color="Black", width=1)))

    # Define the layout of the figure, including the rectangle shape and grid
    layout = go.Layout(shapes=[rectangle] + grid,
                   xaxis=dict(range=[x0 - 0.1 * nelx_value, x1 + 0.1 * nelx_value], autorange=False, showgrid=True, zeroline=True, showline=True, ticks='', showticklabels=True),
                   yaxis=dict(range=[y0 - 0.1 * nely_value, y1 + 0.1 * nely_value], autorange=False, showgrid=True, zeroline=True, showline=True, ticks='', showticklabels=True, scaleanchor = "x", scaleratio = nely_value/nelx_value),
                   showlegend=False,
                   autosize=False,
                   margin=dict(l=20, r=20, b=20, t=20, pad=10),
                   paper_bgcolor="White",)  # Change background color here


    # Create the figure
    fig = go.Figure(layout=layout)

    return fig


app.layout = layout



if __name__ == "__main__":
    app.run_server(debug=True)