import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np

# Generate example matrix
matrix = np.random.random((10, 10))

# Create Dash app with MATERIA theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

# Define the site 1 layout
site_1_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Menu"),
                        html.Button("Preview", id="preview-button", className="preview-button"),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.H2("Matrix Heatmap"),
                        dcc.Graph(
                            id="heatmap-graph",
                            figure=go.Figure(
                                data=go.Heatmap(
                                    z=matrix,
                                    colorscale="Magma",
                                ),
                                layout=go.Layout(
                                    title="Matrix Heatmap",
                                    xaxis=dict(title="X-axis"),
                                    yaxis=dict(title="Y-axis"),
                                    margin=dict(t=40),
                                ),
                            ),
                            style={"height": "500px"},
                        ),
                    ],
                    width=9,
                ),
            ]
        )
    ],
    className="site-content",
)

# Define the site 2 layout
site_2_layout = html.Div(
    id="site-2-content",
    className="site-content",
    children=[
        html.H2("Site 2 Content"),
    ],
)

# Create Dash app layout
app.layout = html.Div(
    children=[
        # Top Navbar
        html.Nav(
            children=[
                html.Img(src="logo.png", alt="Logo", className="logo"),
                html.H1("Topopt at LPL", className="title"),
                dbc.Button("Site 1", id="site-1-button", n_clicks=0, className="navbar-button"),
                dbc.Button("Site 2", id="site-2-button", n_clicks=0, className="navbar-button"),
            ],
            className="navbar",
        ),
        # Mainframe
        html.Div(
            id="mainframe",
            className="mainframe",
            children=[
                html.Div(id="site-content"),
            ],
        ),
    ]
)

# Callback to switch between sites
@app.callback(
    dash.dependencies.Output("site-content", "children"),
    [
        dash.dependencies.Input("site-1-button", "n_clicks"),
        dash.dependencies.Input("site-2-button", "n_clicks"),
    ]
)
def display_site(site1_clicks, site2_clicks):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "site-1-button":
        return site_1_layout
    elif triggered_id == "site-2-button":
        return site_2_layout
    else:
        return site_1_layout

if __name__ == "__main__":
    app.run_server(debug=True)
