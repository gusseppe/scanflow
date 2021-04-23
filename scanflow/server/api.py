import json
import os
import requests
import uvicorn
import logging

from scanflow import tools
from scanflow.deploy.deploy import Deploy
from fastapi import FastAPI, BackgroundTasks

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


app = FastAPI(title='Scanflow API',
              description='Server that allows agents to manipulate the user workflow.',
              )

def trigger_mas(client, mas_port):
    runs_info_inference = client.search_runs("0", "tag.mlflow.runName='inference_batch'",
                                             order_by=["attribute.start_time DESC"],
                                             max_results=1)
    runs_info_train = client.search_runs("0", "tag.mlflow.runName='training'",
                                         order_by=["attribute.start_time DESC"],
                                         max_results=1)

    n_predictions = runs_info_inference[0].data.params['n_predictions']
    x_len = runs_info_train[0].data.params['x_len']
    print(n_predictions, x_len)
    if n_predictions > 0.01 * x_len:
        # if n_predictions > 0.05 * x_len:
        url = f'http://localhost:{mas_port}/send/checker/anomaly'
        response = requests.get(
            url=url,
            headers={"accept": "application/json"})

        response_json = json.loads(response.text)
        print(response_json)

def get_app_fastapi(client, mas_port):

    @app.post("/run/workflow",
              tags=['Running'],
              summary="Run a workflow")
    async def run_workflow(content: dict, background_task: BackgroundTasks):
        background_task.add_task(trigger_mas, client, mas_port)

        paths = tools.get_scanflow_paths(content['app_dir'])
        meta_dir = paths['meta_dir']

        workflows_metadata_name = f"{content['name']}.json"
        workflows_metadata_path = os.path.join(meta_dir, workflows_metadata_name)
        try:
            with open(workflows_metadata_path) as fread:
                setup = json.load(fread)

        except ValueError as e:
            logging.error(f"{e}")
            logging.error(f"[-] Workflow metadata [{content['name']}] does not exist.")

        setup['executors'][0]['parameters'] = content['parameters']

        deployer = Deploy(setup)
        deployer.run_workflows(verbose=True)
        result = deployer.logs_run_workflow[0]['envs'][0]['result']

        response = {"result": result}

        return response


    return app
