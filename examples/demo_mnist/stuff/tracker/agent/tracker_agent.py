
import uvicorn
import os
import json
import mlflow
from typing import Optional, List

from pydantic import BaseModel, HttpUrl
import aiohttp
import logging

from mlflow.tracking import MlflowClient
from fastapi import FastAPI

client = MlflowClient()

class Config():
    agent_name = 'Tracker'

    app_dir = '/home/guess/Desktop/scanflow/examples/demo_mnist/'
    tracker_belief_filename = 'summary.json'
    checker_agent_uri = "http://inference-mnist-checker-agent:8005/checker/anomaly"
    improver_agent_uri = "http://inference-mnist-improver-agent:8006/improver/conclusions"

# consider put this into startup fastapi function

experiment = client.get_experiment_by_name(Config.agent_name)

if experiment:
    experiment_id = experiment.experiment_id
    logging.info(f"[Tracker]  '{Config.agent_name}' experiment loaded.")
else:
    experiment_id = client.create_experiment(Config.agent_name)
    logging.info(f"[Tracker]  '{Config.agent_name}' experiment does not exist. Creating a new one.")

app = FastAPI(title='Tracker Agent API',
              description='Actions and Beliefs for the Tracker Agent')

@app.on_event("startup")
async def startup_event():
    app.aiohttp_session = aiohttp.ClientSession()

@app.on_event("shutdown")
async def shutdown_event():
    await app.aiohttp_session.close()

class Feedback():
    url: str
    name: str

class Receiver():
    name: str
    address: str #HttpUrl

class Message(object):
    def __init__(self,
                 content: dict,
                 performative: str,
                 receiver: str):

        self.content = content
        self.performative = performative
        self.receiver = receiver


@app.get("/send/checker/anomaly",
         tags=['Actions'],
         summary="Send input to the anomaly detector")
async def send_to_checker():
    # runs_info = client.list_run_infos('0', # Get the default experiment
    #                                   order_by=["attribute.start_time DESC"])

    runs_info_training = client.search_runs('0', "tag.mlflow.runName='training'",
                                   order_by=["attribute.start_time DESC"],
                                   max_results=1)

    runs_info_inference = client.search_runs('0', "tag.mlflow.runName='inference_batch'",
                       order_by=["attribute.start_time DESC"],
                       max_results=1)

    if runs_info_inference and runs_info_training:
        # last_run_id = runs_info_inference[0].run_id
        last_run_inference_id = runs_info_inference[0].info.run_id
        x_inference_artifact = "x_inference.npy"
        y_inference_artifact = "y_inference.npy"

        x_train_artifact = "x_train.npy"
        y_train_artifact = "y_train.npy"

        last_run_training_id = runs_info_training[0].info.run_id
        print(last_run_inference_id, last_run_training_id)

        # Get the feedback from Checker
        content = {"run_inference_id":last_run_inference_id,
                   "x_inference_artifact": x_inference_artifact,
                   "y_inference_artifact": y_inference_artifact,

                   "run_training_id":last_run_training_id,
                   "x_train_artifact": x_train_artifact,
                   "y_train_artifact": y_train_artifact,

                   }
        message = Message(content, "INFORM", Config.checker_agent_uri)
        async with app.aiohttp_session.post(message.receiver, json=message.content) as response:
            result_checker = await response.json(content_type=None)

        # Send the feedback to Improver
        content = result_checker['feedback']
        message = Message(content, "INFORM", Config.improver_agent_uri)
        async with app.aiohttp_session.post(message.receiver, json=message.content) as response:
            result_improver = await response.json(content_type=None)

        response = {'feedback': result_checker['feedback'],
                    'conclusions': result_improver['conclusions']}

        with open(Config.tracker_belief_filename, 'w') as fout:
            json.dump(response, fout)

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=Config.agent_name) as mlrun:
            mlflow.log_artifact(Config.tracker_belief_filename, 'Summary')
            mlflow.log_param(key='response_time',
                             value=1.2)
    else:
        response = {"Result": 'No input found'}

    return response


@app.get("/send/checker/human",
         tags=['Actions'],
         summary="Send input to a human")
async def send_to_checker():
    runs_info = client.list_run_infos('0', # Default
                                      order_by=["attribute.start_time DESC"])

    if runs_info:
        last_run_id = runs_info[0].run_id
        input_artifact = os.path.join("Input", "input.csv")

        content = {"run_id":last_run_id, "input": input_artifact}
        message = Message(content, "INFORM", Config.checker_agent_uri)

        async with app.aiohttp_session.post(Config.checker_agent_uri, json=message.content) as response:
            result = await response.json(content_type=None)

        response = result['feedback']

    else:
        response = {"Result": 'No input found'}

    return response

@app.get("/tracker/model/current",
         tags=['Beliefs'],
         summary="Get current deployed model")
async def get_current_model(model_name='mnist_cnn'):
    models = client.search_model_versions(f"name='{model_name}'")
    current_model = None
    for model in models:
        if model.current_stage == 'Production':
            current_model = model
            break

    if current_model is None:
       current_model = models[-1]

    response = {"model": current_model}

    return response

@app.get("/tracker/model/new",
         tags=['Beliefs'],
         summary="Get new trained model")
async def get_new_model(model_name='mnist_cnn_new'):
    models = client.search_model_versions(f"name='{model_name}'")
    response = {"model": models[-1]}

    return response

@app.get("/tracker/last/training",
         tags=['Beliefs'],
         summary="Get last training data")
async def get_current_model():

    models = client.search_model_versions("name='mnist_cnn'")
    response = {"model": models[-1]}

    return response

@app.get("/tracker/last/testing",
         tags=['Beliefs'],
         summary="Get current deployed model")
async def get_current_model():

    models = client.search_model_versions("name='mnist_cnn'")
    response = {"model": models[-1]}

    return response

@app.get("/tracker/last/inference",
         tags=['Beliefs'],
         summary="Get current deployed model, that is, in Production.")
async def get_current_model():

    models = client.search_model_versions("name='mnist_cnn'")
    current_model = None
    for model in models:
        if model.current_stage == 'Production':
            current_model = model
            break

    response = {"model": current_model}

    return response
