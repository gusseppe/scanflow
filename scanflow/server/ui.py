import dash
import json
import os
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import mlflow
from mlflow.tracking import MlflowClient

tracker_uri = "http://0.0.0.0:8002"
mlflow.set_tracking_uri(tracker_uri)
client = MlflowClient()

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

def set_card_content(ctn_type, ctn_name, url=None):
    if url:
        card_content = [
            html.B(dbc.CardHeader(f"{ctn_type}")),
            dbc.CardBody(
                [
                    html.H5(f"{ctn_name}", className="card-title"),
                    dcc.Link(href=url),
                    # dcc.Link(href='http://localhost:8002'),
                    # html.P(
                    #     "This is some card content that we'll reuse",
                    #     className="card-text",
                    # ),
                ]
            ),
        ]
    else:
        card_content = [
            html.B(dbc.CardHeader(f"{ctn_type}")),
            dbc.CardBody(
                [
                    html.H5(f"{ctn_name}", className="card-title"),
                ]
            ),
        ]

    return card_content

def set_cards():
    experiment = client.get_experiment_by_name("Scanflow")
    experiment_id = experiment.experiment_id


    runs_info = client.search_runs(experiment_id, "tag.mlflow.runName='containers'",
                                   order_by=["attribute.start_time DESC"],
                                   max_results=1)
    containers_metadata_path = runs_info[0].data.params['path']
    with open(containers_metadata_path) as fread:
        containers_info_loaded = json.load(fread)

    cards = list()
    for container_info in containers_info_loaded:
        ctn_name = container_info['name']
        ctn_type = container_info['type']
        if 'port' in container_info.keys():
            if ctn_type == 'agent':
                url = f"http://localhost:{container_info['port']}/docs"
            else:
                url = f"http://localhost:{container_info['port']}"

            card_content = set_card_content(ctn_type.capitalize(), ctn_name, url)
        else:
            card_content = set_card_content(ctn_type.capitalize(), ctn_name)

        card = dbc.Col(dbc.Card(card_content, color="success", outline=True))
        cards.append(card)

    rows = list()
    if len(cards) > 4:
        row1 = dbc.Row(
            cards[:4],
            className="mb-4",
        )
        row2 = dbc.Row(
            cards[4:],
            className="mb-4",
        )
        rows.append(row1)
        rows.append(row2)
    else:
        row = dbc.Row(
            cards,
            className="mb-4",
        )
        rows.append(row)

    return rows

# row_2 = dbc.Row(
#     [
#         dbc.Col(dbc.Card(card_content, color="success", outline=True)),
#         dbc.Col(dbc.Card(card_content, color="warning", outline=True)),
#         dbc.Col(dbc.Card(card_content, color="danger", outline=True)),
#     ],
#     className="mb-4",
# )

rows = set_cards()
cards = html.Div(rows)

sidebar = html.Div(
    [
        html.H2("Scanflow", className="display-4"),
        html.Hr(),
        html.P(
            "Dashboard for user workflows and agents.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Workflows", href="/page-1", active="exact"),
                dbc.NavLink("Agents", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
navbar = dbc.NavbarSimple(
    children=[
        # dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        dbc.DropdownMenu(
            children=[
                # dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Option 1", href="#"),
                dbc.DropdownMenuItem("Option 2", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="Options",
        ),
    ],
    brand="Scanflow",
    brand_href="#",
    color="primary",
    dark=True,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app_dash = dash.Dash(name='Scanflow', external_stylesheets=[dbc.themes.YETI])
app_dash.title = 'Scanflow'

app_dash.layout = html.Div([navbar, dcc.Location(id="url"), sidebar, content])

@app_dash.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return cards
        # return html.P("Here you will find basic information about the system.")
    elif pathname == "/page-1":
        return html.P("This will contain all the deployed workflows.")
    elif pathname == "/page-2":
        return html.P("This will contain information about your agents.")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
