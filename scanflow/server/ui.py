import dash
import json
from scanflow.server import agent
import os
import sys
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from textwrap import dedent
import mlflow
from mlflow.tracking import MlflowClient


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
                    dcc.Link(href=url, target='_blank'),
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

def layout_rows(cards):

    rows = list()
    step = 3
    if len(cards) > step:
        for i in range(0, len(cards), step):
            row = dbc.Row(
                cards[i:i+step],
                justify="center",
                className="mb-4",
            )
            rows.append(row)
    else:
        row = dbc.Row(
            cards,
            className="mb-4",
        )
        rows.append(row)

    return rows

def supervisor_explanation(client):
    agent_name = "Supervisor"
    data = agent.get_metadata(experiment_name=agent_name,
                                     executor_name=agent_name,
                                     client=client)

    # x_train_len = data.params['x_train_len']
    # x_inference_len = data.params['x_inference_len']
    # training_executor_name = data.params['training_executor_name']
    # inference_executor_name = data.params['inference_executor_name']

    if data:
        x_train_len = data.params['x_train_len']
        x_inference_len = data.params['x_inference_len']
        training_executor_name = data.params['training_executor_name']
        inference_executor_name = data.params['inference_executor_name']

        card_content = [
            html.H4(html.B(dbc.CardHeader(f"Last explanation : {agent_name}"))),
            dbc.CardBody(
                [

                    html.H5(f"Action", className="card-title"),
                    html.P(f"Multi Agent activated because of [{inference_executor_name}] executor.", className="card-text"),
                    html.P(f"Got new inference data with {x_inference_len} instances.", className="card-text"),

                    html.H5(f"Result", className="card-title"),
                    html.P(f"Send metadata: train data = {x_train_len} instances to Checker.",
                           className="card-text"),
                    html.P(f"Send inference data = {x_inference_len} instances to Checker.",
                           className="card-text"),
                    html.P(f"Send training executor name = [{training_executor_name}] instances to Checker.",
                           className="card-text")
                ]
            ),
        ]
        card = dbc.Col(dbc.Card(card_content, color="primary", inverse=True),
                       width=6)
    else:
        return None

    return card

def checker_explanation(client):
    agent_name = "Checker"
    # n_anomalies = agent.get_metadata(experiment_name=agent_name,
    #                                 executor_name=agent_name,
    #                                 param='n_anomalies',
    #                                 client=client)
    # x_chosen_len = agent.get_metadata(experiment_name=agent_name,
    #                                  executor_name=agent_name,
    #                                  param='x_chosen_len',
    #                                  client=client)
    # x_new_train_len = agent.get_metadata(experiment_name=agent_name,
    #                                   executor_name=agent_name,
    #                                   param='x_new_train_len',
    #                                   client=client)

    data = agent.get_metadata(experiment_name=agent_name,
                              executor_name=agent_name,
                              client=client)
    if data:
        n_anomalies = data.params['n_anomalies']
        x_chosen_len = data.params['x_chosen_len']
        x_new_train_len = data.params['x_new_train_len']

        x_train = int(x_new_train_len) - int(x_chosen_len)


        card_content = [
            html.H4(html.B(dbc.CardHeader(f"Last explanation : {agent_name}"))),
            dbc.CardBody(
                [

                    html.H5(f"Action", className="card-title"),
                    html.P(f"Received x_train: {x_train} instances from Supervisor.", className="card-text"),
                    html.P(f"Detected anomalies: {n_anomalies} instances.", className="card-text"),
                    html.P(f"Selected anomalies (x_chosen): {x_chosen_len} instances.", className="card-text"),
                    html.P(f"Augmented data (x_train+x_chosen) : {x_new_train_len} instances.", className="card-text"),

                    html.H5(f"Result", className="card-title"),
                    html.P(f"Send augmented data ({x_new_train_len} instances) to Improver", className="card-text")
                ]
            ),
        ]
        card = dbc.Col(dbc.Card(card_content, color="primary", inverse=True),
                       width=6)
    else:
        return None

    return card

def improver_explanation(client):
    agent_name = "Improver"

    # action = agent.get_metadata(experiment_name=agent_name,
    #                                      executor_name=agent_name,
    #                                      param='action',
    #                                      client=client)
    # result = agent.get_metadata(experiment_name=agent_name,
    #                             executor_name=agent_name,
    #                             param='result',
    #                             client=client)
    #
    # p_anomalies = agent.get_metadata(experiment_name=agent_name,
    #                             executor_name=agent_name,
    #                             param='p_anomalies',
    #                             client=client)
    data = agent.get_metadata(experiment_name=agent_name,
                              executor_name=agent_name,
                              client=client)
    if data:
        action = data.params['action']
        result = data.params['result']
        p_anomalies = data.params['p_anomalies']

        # try:
        card_content = [
            html.H4(html.B(dbc.CardHeader(f"Last explanation : {agent_name}"))),
            dbc.CardBody(
                [

                    html.H4(f"Action", className="card-title"),
                    html.P(f"{action}", className="card-text"),
                    html.P(f"p_anomalies = {p_anomalies}", className="card-text"),
                    # html.P(f"{conclusion['action']}", className="card-text"),

                    # html.H4(f"Reason", className="card-title"),
                    # html.P(f"Percentage anomalies > threshold (10%)", className="card-text"),

                    html.H4(f"Result", className="card-title"),
                    html.P(f"{result}", className="card-text"),
                ]
            ),
        ]
        card = dbc.Col(dbc.Card(card_content, color="success", inverse=True),
                       width=6)
        # except:
        #     return None
    else:
        return None

    return card


def planner_explanation(client):
    agent_name = "Planner"

    data = agent.get_metadata(experiment_name=agent_name,
                                executor_name=agent_name,
                                client=client)
    #  = agent.get_metadata(experiment_name=agent_name,
    #                            executor_name=agent_name,
    #                            param='order',
    #                            client=client)
    #
    # experiment = client.get_experiment_by_name(agent_name)
    # experiment_id = experiment.experiment_id
    #
    #
    # runs_info = client.search_runs(experiment_id, f"tag.mlflow.runName='{agent_name}'",
    #                                order_by=["attribute.start_time DESC"],
    #                                max_results=1)

    if data:
        response = data.params
        # response = runs_info[0].data.params

        card_content = [
            html.H4(html.B(dbc.CardHeader(f"Last explanation : {agent_name}"))),
            dbc.CardBody(
                [

                    html.H5(f"Order", className="card-title"),
                    html.P(f"{response['order']}", className="card-text"),
                    html.H5(f"Current model", className="card-title"),
                    html.P(f"{response['current_model_name']}/{response['current_model_version']}", className="card-text"),
                    html.H5(f"New model", className="card-title"),
                    html.P(f"{response['new_model_name']}/{response['new_model_version']}", className="card-text"),
                    html.H5(f"Result", className="card-title"),
                    html.P(f"{response['result']}", className="card-text"),
                    html.P(f"Please consider using this new model.", className="card-text")
                ]
            ),
        ]
        card = dbc.Col(dbc.Card(card_content, color="success", inverse=True),
                       width=6)
    else:
        return None
    # row = dbc.Row(
    #     card,
    #     className="mb-4",
    # )

    return card

def get_explanations(client):
    card1 = supervisor_explanation(client)
    card2 = checker_explanation(client)
    card3 = improver_explanation(client)
    card4 = planner_explanation(client)

    row1 = dbc.Row([card1, card2],
        className="mb-4",
    )
    row2 = dbc.Row([card3, card4],
                  className="mb-4",
                  )

    return [row1, row2]


def set_cards(client):
    experiment = client.get_experiment_by_name("Scanflow")
    experiment_id = experiment.experiment_id


    runs_info = client.search_runs(experiment_id, "tag.mlflow.runName='containers'",
                                   order_by=["attribute.start_time DESC"],
                                   max_results=1)
    containers_metadata_path = runs_info[0].data.params['path']
    try:
        with open(containers_metadata_path) as fread:
            containers_info_loaded = json.load(fread)
    except:
        return None, None, None

    cards = list()
    executor_cards = list()
    agent_cards = list()
    for container_info in containers_info_loaded:
        ctn_name = container_info['name']
        ctn_type = container_info['type']
        if 'port' in container_info.keys():
            if ctn_type == 'agent':
                url = f"http://localhost:{container_info['port']}/docs"
                card_content = set_card_content(ctn_type.capitalize(), ctn_name, url)
                card = dbc.Col(dbc.Card(card_content, color="light", outline=False), width=4)
                agent_cards.append(card)
            else: #mlflow
                url = f"http://localhost:{container_info['port']}"
                card_content = set_card_content(ctn_type.capitalize(), ctn_name, url)
                card = dbc.Col(dbc.Card(card_content, color="success", outline=True), width=4)

        else: #executors
            card_content = set_card_content(ctn_type.capitalize(), ctn_name)
            card = dbc.Col(dbc.Card(card_content, color="info", outline=True), width=4)
            executor_cards.append(card)

        # card = dbc.Col(dbc.Card(card_content, color="success", outline=True), width=4)
        cards.append(card)

    rows = layout_rows(cards)
    cards = html.Div(rows)

    executor_rows = layout_rows(executor_cards)
    executor_cards = html.Div(executor_rows)

    agent_rows = layout_rows(agent_cards)
    rows = get_explanations(client)
    agent_rows.extend(rows)
    agent_cards = html.Div(agent_rows)

    return cards, executor_cards, agent_cards

sidebar = html.Div(
    [
        html.B(html.H2("Scanflow", className="display-4")),
        html.Hr(),
        html.P(
            "UI dashboard. /docs for API.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href=f"/", active="exact"),
                dbc.NavLink("Executors", href="/executors", active="exact"),
                dbc.NavLink("Agents", href="/agents", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# row_2 = dbc.Row(
#     [
#         dbc.Col(dbc.Card(card_content, color="success", outline=True)),
#         dbc.Col(dbc.Card(card_content, color="warning", outline=True)),
#         dbc.Col(dbc.Card(card_content, color="danger", outline=True)),
#     ],
#     className="mb-4",
# )

def get_app_dash(client, mlflow_port, server_port):
    app_dash = dash.Dash(name='Scanflow', external_stylesheets=[dbc.themes.YETI])
    app_dash.title = 'Scanflow'
    content = html.Div(id="page-content", style=CONTENT_STYLE)

    if client:
        cards, executor_cards, agent_cards = set_cards(client)
        mlflow_item = dbc.DropdownMenuItem("Tracker-mlflow",
                             href=f"http://localhost:{mlflow_port}",
                             target='_blank')
    else:
        mlflow_item = dbc.DropdownMenuItem("Tracker-mlflow",
                                           href=f"/",
                                           target='_blank')
        cards = None
        executor_cards = None
        agent_cards = None

    navbar = dbc.NavbarSimple(
        children=[
            # dbc.NavItem(dbc.NavLink("Page 1", href="#")),
            dbc.DropdownMenu(
                children=[
                    # dbc.DropdownMenuItem("More pages", header=True),
                    mlflow_item,
                    dbc.DropdownMenuItem("Scanflow API",
                                         href=f"http://localhost:{server_port}/docs",
                                         target='_blank'),
                ],
                nav=True,
                in_navbar=True,
                label="Quick links",
            ),
        ],
        brand="Scanflow",
        brand_href="#",
        color="primary",
        # dark=True,
    )

    app_dash.layout = html.Div([navbar, dcc.Location(id="url"), sidebar, content])

    @app_dash.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/":
            return cards
            # return html.P("Here you will find basic information about the system.")
        elif pathname == "/executors":
            return executor_cards
            # return html.P("This will contain all the deployed workflows.")
        elif pathname == "/agents":
            return agent_cards
            # return html.P("This will contain information about your agents.")
        # If the user tries to reach a different page, return a 404 message
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )

    return app_dash
