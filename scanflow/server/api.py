import json
import os
import requests
import uvicorn
import logging
import threading
import time

from scanflow import tools
from scanflow.deploy.deploy import Deploy
from fastapi import FastAPI, BackgroundTasks

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


app = FastAPI(title='Scanflow API',
              description='Server that allows agents to manipulate the user workflow.',
              )


# class ThreadingExample(object):
#     """ Threading example class
#     The run() method will be started and it will run in the background
#     until the application exits.
#     """
#
#     def __init__(self, client, mas_port, interval=1):
#         """ Constructor
#         :type interval: int
#         :param interval: Check interval, in seconds
#         """
#         self.client = client
#         self.mas_port = mas_port
#         self.interval = interval
#
#         thread = threading.Thread(target=self.run, args=())
#         thread.daemon = True                            # Daemonize thread
#         thread.start()                                  # Start the execution
#
#     def run(self):
#         """ Method that runs forever """
#         # while True:
#         trigger_mas(self.client, self.mas_port)
            # Do something
            # print('Doing something imporant in the background')
            #
            # time.sleep(self.interval)

def trigger_mas(client, mas_port):
    print("helasda"*10)
    runs_info_inference = client.search_runs("0", "tag.mlflow.runName='inference_batch'",
                                             order_by=["attribute.start_time DESC"],
                                             max_results=1)
    runs_info_train = client.search_runs("0", "tag.mlflow.runName='training'",
                                         order_by=["attribute.start_time DESC"],
                                         max_results=1)

    n_predictions = int(runs_info_inference[0].data.params['n_predictions'])
    x_len = int(runs_info_train[0].data.params['x_len'])
    if n_predictions > 0.01 * x_len:
        print("Starting trigger_mas")
        # if n_predictions > 0.05 * x_len:
        url = f'http://localhost:{mas_port}/send/checker/anomaly'
        response = requests.get(
            url=url,
            headers={"accept": "application/json"})

        response_json = json.loads(response.text)
        print(response_json)
    else:
        print("No need to trigger_mas")

    return None

def trigger_mas2(client, mas_port):
    print(client, mas_port)
    print("asss")

def get_app_fastapi(client, mas_port):
    app = FastAPI(title='Scanflow API',
                  description='Server that allows agents to manipulate the user workflow.',
                  )
    if client is None:
        return app

    @app.post("/run/executor",
              tags=['Running'],
              summary="Run an executor")
    async def run_executor(content: dict, background_task: BackgroundTasks):
        # background_task.add_task(trigger_mas2, client, mas_port)
        # example = ThreadingExample(client, mas_port)
        if 'inference-mnist-inference' in content['name']:
            background_task.add_task(trigger_mas, client, mas_port)


        experiment = client.get_experiment_by_name("Scanflow")
        experiment_id = experiment.experiment_id

        executor_name = content['name']
        runs_info = client.search_runs(experiment_id, f"tag.mlflow.runName='{executor_name}'",
                                       order_by=["attribute.start_time DESC"],
                                       max_results=1)
        # runs_info[0].data.params = content['parameters']
        executor = runs_info[0].data.params
        executor['parameters'] = content['parameters']

        deployer = Deploy()
        result = deployer.run_executor(executor)
        # result = deployer.logs_run_workflow[0]['envs'][0]['result']

        response = {"result": result}

        # print(type(trigger_mas(client, mas_port)))

        return response

    return app
    # @app.post("/run/workflow",
    #           tags=['Running'],
    #           summary="Run a workflow")
    # async def run_workflow(content: dict, background_task: BackgroundTasks):
    #     background_task.add_task(trigger_mas, client, mas_port)
    #
    #     paths = tools.get_scanflow_paths(content['app_dir'])
    #     meta_dir = paths['meta_dir']
    #
    #     workflows_metadata_name = f"{content['name']}.json"
    #     workflows_metadata_path = os.path.join(meta_dir, workflows_metadata_name)
    #     try:
    #         with open(workflows_metadata_path) as fread:
    #             setup = json.load(fread)
    #
    #     except ValueError as e:
    #         logging.error(f"{e}")
    #         logging.error(f"[-] Workflow metadata [{content['name']}] does not exist.")
    #
    #     setup['executors'][0]['parameters'] = content['parameters']
    #
    #     deployer = Deploy(setup)
    #     deployer.run_workflows(verbose=True)
    #     result = deployer.logs_run_workflow[0]['envs'][0]['result']
    #
    #     response = {"result": result}
    #
    #     return response

