import os
import mlflow
import json
import aiohttp
import logging
import tools
import urllib.parse

from mlflow.tracking import MlflowClient
from mlflow.exceptions import  MlflowException
from typing import Optional, List, Dict

from fastapi import FastAPI, Response, Request, UploadFile
from pydantic import BaseModel, HttpUrl

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


# tracker_uri = "http://tracker-agent-mnist:8003/tracker/current/model"
# checker_uri = "http://checker-agent-mnist:8005/feedback/anomaly/last"
# planner_uri = "http://planner-agent-mnist:8007/planner/plans"

# agent_name = 'Improver'
# consider put this into startup fastapi function
client = MlflowClient()

class Config():
    agent_name = 'Improver'

    app_dir = '/home/guess/Desktop/scanflow/examples/demo_mnist/'
    improver_filename = 'conclusions.json'
    tracker_uri = "http://inference-mnist-tracker-agent:8003"
    checker_uri = "http://inference-mnist-checker-agent:8005/feedback/anomaly/last"
    planner_uri = "http://inference-mnist-planner-agent:8007/planner/plans"

    host_uri = 'http://192.168.96.1:8050/run/workflow'

experiment = client.get_experiment_by_name(Config.agent_name)

if experiment:
    experiment_id = experiment.experiment_id
    logging.info(f"[Improver]  '{Config.agent_name}' experiment loaded.")
else:
    experiment_id = client.create_experiment(Config.agent_name)
    logging.info(f"[Improver]  '{Config.agent_name}' experiment does not exist. Creating a new one.")


app = FastAPI(title='Improver Agent API',
              description='Actions and Beliefs for the Improver Agent',
              )

@app.on_event("startup")
async def startup_event():
    app.aiohttp_session = aiohttp.ClientSession()

@app.on_event("shutdown")
async def shutdown_event():
    await app.aiohttp_session.close()

class Message(object):
    def __init__(self,
                 content: dict,
                 performative: str,
                 receiver: str):

        self.content = content
        self.performative = performative
        self.receiver = receiver


@app.post("/improver/conclusions",
          tags=['Actions'],
          summary="Call improver to get conclusions")
async def execute_improver(feedback: dict):

    n_anomalies = feedback['n_anomalies']
    p_anomalies = feedback['percentage_anomalies']

    if p_anomalies <= 0.05:
        response = {'conclusions': f'Normal behavior!, {p_anomalies}% anomalies'}
    elif 0.05 < p_anomalies < 0.1:
        response = {'conclusions': f'Alert!, {p_anomalies}% anomalies'}
    else:
        # Get the current model from tracker
        relative_uri = "tracker/model/current"
        uri = urllib.parse.urljoin(Config.tracker_uri, relative_uri)
        message = Message("", "INFORM", uri)
        async with app.aiohttp_session.get(message.receiver) as response:
            result_tracker = await response.json(content_type=None)


        new_model_name = 'mnist_cnn_new'
        scanflow_request = {
            'app_dir': Config.app_dir,
            'name': 'retraining-mnist',
            'parameters':{
                'run_id': feedback['checker_agent_run_id'],
                'model_name': new_model_name,
                'x_new_train_artifact': feedback['x_new_train_artifact'],
                'y_new_train_artifact': feedback['y_new_train_artifact'],
                'x_test_path': './mnist/test_images.npy',
                'y_test_path': './mnist/test_labels.npy',
            },
        }
        async with app.aiohttp_session.post(Config.host_uri, json=scanflow_request) as response:
            result_host = await response.json(content_type=None)

        print(result_host)

        old_metric = tools.get_metadata(executor_name='training', metric='accuracy')
        new_metric = tools.get_metadata(executor_name='retraining', metric='accuracy')

        if new_metric > old_metric:
            relative_uri = "tracker/model/new"
            uri = urllib.parse.urljoin(Config.tracker_uri, relative_uri)
            message = Message("", "INFORM", uri)
            async with app.aiohttp_session.get(message.receiver) as response:
                tracker_new_model = await response.json(content_type=None)
            # Communicate with the Planner to do replaced the old model by the newer one
            content = {'conclusions': {
                            'order': 'Transition',
                            'current_model_name': result_tracker['model']['name'],
                            'current_model_version': result_tracker['model']['version'],
                            'new_model_name': new_model_name,
                            'new_model_version': tracker_new_model['model']['version'],
                        }
                      }
            message = Message(content, "INFORM", Config.planner_uri)
            async with app.aiohttp_session.post(message.receiver, json=content) as response:
                result_planner = await response.json(content_type=None)

            response = {'conclusions': {
                    "action": f"Retraining the model using the new augmented data={feedback['x_new_train_artifact']}",
                    "reason":  f"current_metric={new_metric} < new_metric={new_metric}",
                    "planner": result_planner,
                }
            }
        else:
            response = {'conclusions': {
                    "action": f'No retraining triggered',
                    "reason":  f"current_metric={new_metric} > new_metric={new_metric}"
                }
            }



        # Get the input data from checker
        # message = Message("", "INFORM", Config.checker_uri)
        # async with app.aiohttp_session.get(message.receiver) as response:
        #     result_checker = await response.json(content_type=None)
        # feedback = result_checker['feedback']
        # client.download_artifacts(feedback['input_run_id'],
        #                       feedback['input_path'],
        #                       '/tmp/')
        #
        # input_local_path = os.path.join('/tmp', feedback['input_path'])

        # The retraining begins here

        # Prepare the message

        #call to host server
        # request = {'app_dir': Config.app_dir,
        #            'name': 'retraining-mnist', # workflow name
        #            # 'name': 'retraining-mnist', # workflow name
        #            'parameters': {'run_id': ''}}
        # async with app.aiohttp_session.post(Config.host_uri, json=request) as response:
        #     result_host = await response.json(content_type=None)
        #
        # print(result_host)

        # new_model_name = f"{result_tracker['model']['name']}_new"
        # print(new_model_name)
        # class AddN(mlflow.pyfunc.PythonModel):
        #
        #     def __init__(self, n):
        #         self.n = n
        #
        #     def predict(self, context, model_input):
        #         return model_input.apply(lambda column: column + self.n)
        #
        # new_model = AddN(n=5)
        #
        # with mlflow.start_run(experiment_id=experiment_id,
        #                       run_name=Config.agent_name) as mlrun:
        #     mlflow.pyfunc.log_model(
        #         python_model=new_model,
        #         artifact_path=new_model_name,
        #         registered_model_name=new_model_name
        #     )
        # The retraining ends here

        # Communicate with the Planner
        # content = {'conclusions': {
        #                 'order': 'Transition new model to Production.',
        #                 'current_model_name': result_tracker['model']['name'],
        #                 'current_model_version': result_tracker['model']['version'],
        #                 'new_model_name': new_model_name,
        #             }
        #           }
        # message = Message(content, "INFORM", Config.planner_uri)
        # async with app.aiohttp_session.post(message.receiver, json=content) as response:
        #     result_planner = await response.json(content_type=None)

    with open(Config.improver_filename, 'w') as fout:
        json.dump(response, fout)

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=Config.agent_name) as mlrun:
        mlflow.log_artifact(Config.improver_filename, 'Conclusions')

    return  response

@app.get("/improver/conclusions/last",
         tags=['Beliefs'],
         summary='Get last Improver conclusions')
async def get_last_conclusions():
    runs_info = client.list_run_infos(experiment_id,
                                      order_by=["attribute.start_time DESC"])
    if runs_info:
        last_run_id = runs_info[0].run_id
        conclusions_artifact_path = os.path.join('Conclusions', 'conclusions.json')

        try:
            client.download_artifacts(last_run_id,
                                      conclusions_artifact_path,
                                      '/tmp/')
        except:
            response = {"feedback": 'No conclusions found yet'}
            return response

        conclusions_local_path = os.path.join('/tmp', conclusions_artifact_path)
        with open(conclusions_local_path) as fread:
            conclusions = json.load(fread)

        response = {"conclusions": conclusions}
    else:
        response = {"conclusions": 'No experiments yet'}

    return response

