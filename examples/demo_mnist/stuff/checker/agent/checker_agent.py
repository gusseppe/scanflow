import uvicorn
import agent
import numpy as np
import os
import mlflow
import json
import pandas as pd
import logging
import aiohttp
from mlflow.tracking import MlflowClient
from mlflow.exceptions import  MlflowException
from typing import Optional, List, Dict

from fastapi import FastAPI, Response, Request, UploadFile
from pydantic import BaseModel, HttpUrl
logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


agent_name = 'Checker'
# consider put this into startup fastapi function
client = MlflowClient()

class Config():
    agent_name = 'Checker'

    app_dir = agent.get_app_dir()
    # app_dir = '/home/guess/Desktop/scanflow/examples/demo_mnist/'
    host_uri = agent.get_host_uri()
    host_uri = f"{host_uri}/run/executor"
    # host_uri = 'http://192.168.96.1:8050/run/executor'
    # host_uri = 'http://192.168.96.1:8050/run/workflow'

experiment = client.get_experiment_by_name(agent_name)

if experiment:
    experiment_id = experiment.experiment_id
    logging.info(f"[Checker]  '{agent_name}' experiment loaded.")
else:
    experiment_id = client.create_experiment(agent_name)
    logging.info(f"[Checker]  '{agent_name}' experiment does not exist. Creating a new one.")



app = FastAPI(title='Checker Agent API',
              description='Actions and Beliefs for the Checker Agent',
              )

@app.on_event("startup")
async def startup_event():
    app.aiohttp_session = aiohttp.ClientSession()

@app.on_event("shutdown")
async def shutdown_event():
    await app.aiohttp_session.close()

class Feedback(BaseModel):
    url: str
    name: str

class Receiver(BaseModel):
    name: str
    address: str #HttpUrl

class Message(BaseModel):
    performative: str
    content: str
    # receiver: Receiver
    # content: List[Feedback] = []



async def detector(scanflow_request):

    # Call the detector-mnist workflow in local
    request = {'app_dir': scanflow_request['app_dir'],
               'name': scanflow_request['name'], # workflow name
               'parameters': scanflow_request['parameters']}

    async with app.aiohttp_session.post(Config.host_uri, json=request) as response:
        result_host = await response.json(content_type=None)

    print(result_host)

    # Search for the latest detector-mnist experiment
    runs_info = client.search_runs('0', "tag.mlflow.runName='detector'",
                                   order_by=["attribute.start_time DESC"],
                                   max_results=1)

    last_run_id = runs_info[0].info.run_id
    x_chosen_artifact_path = "x_inference_chosen.npy"
    y_chosen_artifact_path = "y_inference_chosen.npy"

    x_inference_anomalies_path = "E_test.csv"

    client.download_artifacts(last_run_id,
                              x_chosen_artifact_path,
                              '/tmp/')
    client.download_artifacts(last_run_id,
                              y_chosen_artifact_path,
                              '/tmp/')

    client.download_artifacts(last_run_id,
                              x_inference_anomalies_path,
                              '/tmp/')

    x_chosen_path = os.path.join('/tmp', x_chosen_artifact_path)
    y_chosen_path = os.path.join('/tmp', y_chosen_artifact_path)
    x_inference_anomalies_path = os.path.join('/tmp', x_inference_anomalies_path)

    x_chosen = np.load(x_chosen_path)
    y_chosen = np.load(y_chosen_path)
    x_inference_anomalies = pd.read_csv(x_inference_anomalies_path)

    # if len(x_inference) < 1000: #Simulate corrupted data
    #     anomalies = np.random.choice([0, 1],
    #                                  size=(len(x_inference),),
    #                                  p=[0.8, 0.2])
    # else:
    #     anomalies = np.random.choice([0, 1],
    #                                  size=(len(x_inference),),
    #                                  p=[0.98, 0.02])

    return await (x_chosen, y_chosen, x_inference_anomalies)
    # return x_chosen, y_chosen, x_inference_anomalies, last_run_id

@app.post("/checker/anomaly",
          tags=['Actions'],
          summary="Call anomaly detector")
async def execute_checker_anomaly(content: Dict[str, str]):

    client.download_artifacts(content['run_training_id'],
                              content['x_train_artifact'],
                              '/tmp/')
    client.download_artifacts(content['run_training_id'],
                              content['y_train_artifact'],
                              '/tmp/')

    # x_inference_path = os.path.join('/tmp', content['x_inference_artifact'])
    # y_inference_path = os.path.join('/tmp', content['y_inference_artifact'])

    x_train_path = os.path.join('/tmp', content['x_train_artifact'])
    y_train_path = os.path.join('/tmp', content['y_train_artifact'])

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)

    scanflow_request = {
        # 'app_dir': Config.app_dir,
        'name': 'detector-inference-mnist-detector-batch',
        # 'name': 'detector-mnist',
        'parameters':{
            'run_training_id': content['run_training_id'],
            'run_inference_id': content['run_inference_id'],
            'x_train_artifact': content['x_train_artifact'],
            'x_inference_artifact': content['x_inference_artifact'],
            'y_inference_artifact': content['y_inference_artifact']
        },
    }
    # request = {'app_dir': scanflow_request['app_dir'],
    #            'name': scanflow_request['name'], # workflow name
    #            'parameters': scanflow_request['parameters']}

    async with app.aiohttp_session.post(Config.host_uri, json=scanflow_request) as response:
        result_host = await response.json(content_type=None)

    print(result_host)

    # Search for the latest detector-mnist experiment
    runs_info_detector = client.search_runs('0', "tag.mlflow.runName='detector'",
                                   order_by=["attribute.start_time DESC"],
                                   max_results=1)

    last_run_detector_id = runs_info_detector[0].info.run_id
    x_chosen_artifact_path = "x_inference_chosen.npy"
    y_chosen_artifact_path = "y_inference_chosen.npy"

    x_inference_anomalies_path = "E_test.csv"

    client.download_artifacts(last_run_detector_id,
                              x_chosen_artifact_path,
                              '/tmp/')
    client.download_artifacts(last_run_detector_id,
                              y_chosen_artifact_path,
                              '/tmp/')

    client.download_artifacts(last_run_detector_id,
                              x_inference_anomalies_path,
                              '/tmp/')

    x_chosen_path = os.path.join('/tmp', x_chosen_artifact_path)
    y_chosen_path = os.path.join('/tmp', y_chosen_artifact_path)
    x_inference_anomalies_path = os.path.join('/tmp', x_inference_anomalies_path)

    x_chosen = np.load(x_chosen_path)
    y_chosen = np.load(y_chosen_path)
    x_inference_anomalies = pd.read_csv(x_inference_anomalies_path)

    # (x_chosen, y_chosen, x_inference_anomalies) = detector(scanflow_request)

    # print(x_train.shape)
    # print(x_chosen.shape)
    # print("-------")
    # print(y_train.shape)
    # print(y_chosen.shape)
    x_new_train = np.append(x_train, x_chosen, axis=0)
    y_new_train = np.append(y_train, y_chosen, axis=0)

    print(x_new_train.shape)

    with open('x_chosen.npy', 'wb') as f:
        np.save(f, x_chosen)
    with open('y_chosen.npy', 'wb') as f:
        np.save(f, y_chosen)

    with open('x_new_train.npy', 'wb') as f:
        np.save(f, x_new_train)
    with open('y_new_train.npy', 'wb') as f:
        np.save(f, y_new_train)
    # d_anomalies = {"anomalies": detector(x_inference, y_inference)}
    # d_anomalies = {"anomalies": [1, 0, 1, 1, 0]}

    n_anomalies = sum(x_inference_anomalies['Anomaly'])
    p_anomalies = n_anomalies/len(x_inference_anomalies['Anomaly'])
    # n_anomalies = sum(d_anomalies['anomalies'])
    # p_anomalies = sum(d_anomalies['anomalies'])/len(d_anomalies['anomalies'])

    feedback = {
        'inference_run_id': content['run_inference_id'],
        'x_inference_artifact': content['x_inference_artifact'],
        'y_inference_artifact': content['y_inference_artifact'],

        'detector_run_id': last_run_detector_id,
        'x_chosen_artifact': 'x_inference_chosen.npy',
        'y_chosen_artifact': 'x_inference_chosen.npy',

        'x_new_train_artifact': 'x_new_train.npy',
        'y_new_train_artifact': 'y_new_train.npy',

        'n_anomalies': int(n_anomalies),
        'percentage_anomalies': float(p_anomalies)
    }

    feedback_filename = 'feedback_anomaly.json'
    artifact_name = 'Detector'

    df_preds = x_inference_anomalies
    df_preds.to_csv("anomalies.csv", index=False)


    with open(feedback_filename, 'w') as fout:
        json.dump(feedback, fout)

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=agent_name) as mlrun:
        mlflow.log_artifact('anomalies.csv')
        mlflow.log_artifact('x_chosen.npy')
        mlflow.log_artifact('y_chosen.npy')
        mlflow.log_artifact('x_new_train.npy')
        # mlflow.log_artifact('x_new_train.npy', 'Improver')
        mlflow.log_artifact('y_new_train.npy')
        mlflow.log_artifact(feedback_filename)
        mlflow.log_param(key='x_chosen_len',
                         value=f"{len(x_chosen)}")
        mlflow.log_param(key='x_new_train_len',
                         value=f"{len(x_new_train)}")
        mlflow.log_param(key='n_anomalies',
                         value=f"{n_anomalies}")
        mlflow.log_param(key='p_anomalies',
                         value=f"{p_anomalies}")
        mlflow.log_param(key='feedback',
                         value=f"{feedback_filename}")
        # mlflow.log_param(key='feedback',
        #                  value=f"{artifact_name}/{feedback_filename}")

    feedback['checker_agent_run_id'] = mlrun.info.run_id
    print(feedback)
    response = {"feedback": feedback}

    return response

@app.get("/feedback/anomaly/last",
         tags=['Beliefs'],
         summary='Get last anomaly feedback')
async def get_last_feedback():
    runs_info = client.list_run_infos(experiment_id,
                                      order_by=["attribute.start_time DESC"])
    if runs_info:
        last_run_id = runs_info[0].run_id
        feedback_artifact_path = os.path.join('Detector', 'feedback_anomaly.json')

        try:
            client.download_artifacts(last_run_id,
                                      feedback_artifact_path,
                                      '/tmp/')
        except:
            response = {"feedback": 'No anomaly feedback yet'}
            return response

        feedback_local_path = os.path.join('/tmp', feedback_artifact_path)
        with open(feedback_local_path) as fread:
            feedback = json.load(fread)

        response = {"feedback": feedback}
    else:
        response = {"feedback": 'No experiments yet'}

    return response


