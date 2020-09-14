import os
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import dash_daq as daq
import dash_cytoscape as cyto

import pandas as pd

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
df = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "spc_data.csv")))

params = list(df)
max_length = len(df)

def build_tab_1():
    return [
        # Manually select metrics
        html.Div([
            # id="set-specs-intro-container",
            # className='twelve columns',
            html.H4('Workflows'),
            html.Div([
                cyto.Cytoscape(
                    id='cytoscape-compound',
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '550px'},
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                # 'content': 'data(label)',
                                # 'content': 'data(name)',
                                # 'text-valign': 'center',
                                # 'color': 'white',
                                # 'text-outline-width': 2,
                                # 'background-color': '#888'
                            }
                        },
                        {
                            'selector': '.selected',
                            'style': {
                                # 'width': 5,
                                'background-color': 'black',
                                'line-color': 'black',
                                'target-arrow-color': 'black',
                                'source-arrow-color': 'black',
                                'text-outline-color': 'black'
                            },
                        },
                        {
                            'selector': '#workflow1',
                            'style': {
                                # 'width': 5,
                                'content': 'data(label)',
                                'color': 'white',
                                'background-color': 'black',
                                'line-color': 'black',
                                'target-arrow-color': 'black',
                                'source-arrow-color': 'black',
                                'text-outline-color': 'black',
                                # 'width': 40,
                                # 'height': 40,
                            },
                        },
                        {
                            'selector': '#workflow2',
                            'style': {
                                'content': 'data(label)',
                                'color': 'white',
                                'background-color': 'black',
                                'line-color': 'black',
                                'target-arrow-color': 'black',
                                'source-arrow-color': 'black',
                                'text-outline-color': 'black'
                            },
                        },
                        {
                            'selector': '#workflow3',
                            'style': {
                                'content': 'data(label)',
                                'color': 'white',
                                'background-color': 'black',
                                'line-color': 'black',
                                'target-arrow-color': 'black',
                                'source-arrow-color': 'black',
                                'text-outline-color': 'black'
                            },
                        },
                        {
                            'selector': '#datascience',
                            'style': {
                                'content': 'data(label)',
                                'color': 'white',
                                'background-color': 'black',
                                'line-color': 'black',
                                'target-arrow-color': 'black',
                                'source-arrow-color': 'black',
                                'text-outline-color': 'black'
                            },
                        },
                        {
                            'selector': '#dataeng',
                            'style': {
                                'content': 'data(label)',
                                'color': 'white',
                                'background-color': 'black',
                                'line-color': 'black',
                                'target-arrow-color': 'black',
                                'source-arrow-color': 'black',
                                'text-outline-color': 'black'
                            },
                        },
                        {
                            'selector': '#gathering',
                            'style': {
                                # 'width': 5,
                                'content': 'data(name)',
                                'background-color': '#03d7fc',
                                'content': 'data(name)',
                                'text-valign': 'center',
                                'color': 'white',
                                'text-outline-width': 2,
                                # 'background-color': '#888'
                                # 'line-color': 'black',
                                # 'target-arrow-color': 'black',
                                # 'source-arrow-color': 'black',
                                # 'text-outline-color': 'black'
                            },
                        },
                        {
                            'selector': '#preprocessing',
                            'style': {
                                # 'width': 5,
                                'content': 'data(name)',
                                'background-color': '#03d7fc',
                                'content': 'data(name)',
                                'text-valign': 'center',
                                'color': 'white',
                                'text-outline-width': 2,
                                # 'line-color': 'black',
                                # 'target-arrow-color': 'black',
                                # 'source-arrow-color': 'black',
                                # 'text-outline-color': 'black'
                            },
                        },
                        {
                            'selector': '#modeling',
                            'style': {
                                # 'width': 5,
                                'content': 'data(name)',
                                'background-color': '#03d7fc',
                                'content': 'data(name)',
                                'text-valign': 'center',
                                'color': 'white',
                                'text-outline-width': 2,
                            },
                        },
                        {
                            'selector': '#gathering2',
                            'style': {
                                # 'width': 5,
                                'content': 'data(name)',
                                'background-color': '#03d7fc',
                                'content': 'data(name)',
                                'text-valign': 'center',
                                'color': 'white',
                                'text-outline-width': 2,
                            },
                        },
                        {
                            'selector': '#preprocessing2',
                            'style': {
                                # 'width': 5,
                                'content': 'data(name)',
                                'background-color': '#03d7fc',
                                'content': 'data(name)',
                                'text-valign': 'center',
                                'color': 'white',
                                'text-outline-width': 2,
                            },
                        },
                        {
                            'selector': '#predictor',
                            'style': {
                                # 'width': 5,
                                'content': 'data(name)',
                                'background-color': '#03d7fc',
                                'content': 'data(name)',
                                'text-valign': 'center',
                                'color': 'white',
                                'text-outline-width': 2,
                            },
                        },
                        {
                            'selector': '#preprocessing, #modeling',
                            'style': {
                                # 'width': 5,
                                'target-arrow-color': 'blue',
                                'target-arrow-shape': 'vee',
                                'line-color': 'blue'
                                # 'line-color': 'black',
                                # 'target-arrow-color': 'black',
                                # 'source-arrow-color': 'black',
                                # 'text-outline-color': 'black'
                            },
                        },
                    ],
                    elements=[
                        {
                            'data': {'id': 'workflow1',
                                     'label': 'Workflow1',
                                     'parent': 'datascience'}
                        },
                        {
                            'data': {'id': 'workflow2',
                                     'label': 'Workflow2',
                                     'parent': 'datascience'}
                        },
                        {
                            'data': {'id': 'workflow3',
                                     'label': 'Workflow3',
                                     'parent': 'dataeng'}
                        },
                        {
                            'data': {'id': 'datascience', 'label': 'DS team'}
                        },
                        {
                            'data': {'id': 'dataeng', 'label': 'DE team'}
                        },

                        # Children Nodes
                        {
                            'data': {'id': 'gathering',
                                     'name': 'Gathering',
                                     'parent': 'workflow1'},
                            'position': {'x': 100, 'y': 100},
                            # 'classes': 'red' # Single class
                        },
                        {
                            'data': {'id': 'preprocessing',
                                     'name': 'Preprocessing',
                                     'parent': 'workflow1'},
                            'position': {'x': 100, 'y': 200}
                        },
                        {
                            'data': {'id': 'modeling',
                                     'name': 'Modeling',
                                     'parent': 'workflow2'},
                            'position': {'x': 250, 'y': 100}
                        },

                        {
                            'data': {'id': 'gathering2',
                                     'name': 'Gathering',
                                     'parent': 'workflow3'},
                            'position': {'x': 200, 'y': 200},
                            # 'classes': 'red' # Single class
                        },
                        {
                            'data': {'id': 'preprocessing2',
                                     'name': 'Preprocessing',
                                     'parent': 'workflow3'},
                            'position': {'x': 300, 'y': 200}
                        },
                        {
                            'data': {'id': 'predictor',
                                     'name': 'Predictor',
                                     'parent': 'workflow3'},
                            'position': {'x': 450, 'y': 200}
                        },
                        # Edges
                        # {
                        #     'data': {'source': 'can', 'target': 'us'},
                        #     'classes': 'countries'
                        # },
                        {
                            'data': {'source': 'gathering',
                                     'target': 'preprocessing'},
                            'classes': 'cities'
                        },
                        {
                            'data': {'source': 'preprocessing',
                                     'target': 'modeling'},
                            # 'classes': 'cities'
                        },
                        {
                            'data': {'source': 'gathering2',
                                     'target': 'preprocessing2'},
                            # 'classes': 'cities'
                        },
                        {
                            'data': {'source': 'preprocessing2',
                                     'target': 'predictor'},
                            # 'classes': 'cities'
                        }
                    ]
                )
            ])
        ]),
        html.Div(
            id="set-specs-intro-container",
            # className='twelve columns',
            children=html.P(
                "Use historical control limits to establish a benchmark, or set new values."
            ),
        ),
        html.Div(
            id="settings-menu",
            children=[
                html.Div(
                    id="metric-select-menu",
                    # className='five columns',
                    children=[
                        html.Label(id="metric-select-title", children="Select Metrics"),
                        html.Br(),
                        dcc.Dropdown(
                            id="metric-select-dropdown",
                            options=list(
                                {"label": param, "value": param} for param in params[1:]
                            ),
                            value=params[1],
                        ),
                    ],
                ),
                html.Div(
                    id="value-setter-menu",
                    # className='six columns',
                    children=[
                        html.Div(id="value-setter-panel"),
                        html.Br(),
                        html.Div(
                            id="button-div",
                            children=[
                                html.Button("Update", id="value-setter-set-btn"),
                                html.Button(
                                    "View current setup",
                                    id="value-setter-view-btn",
                                    n_clicks=0,
                                ),
                            ],
                        ),
                        html.Div(
                            id="value-setter-view-output", className="output-datatable"
                        ),
                    ],
                ),
            ],
        ),
    ]
