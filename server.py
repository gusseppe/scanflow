import json
import uvicorn
import os
import logging

from scanflow.server import ui
from scanflow.deploy.deploy import Deploy
from scanflow import tools
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

import dash
logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)



# FASTAPI
app = FastAPI(title='Scanflow API',
              description='Server that allows agents to manipulate the user workflow.',
              )

@app.post("/run/workflow",
          tags=['Running'],
          summary="Run a workflow")
async def run_workflow(content: dict):
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

app_dash = ui.app_dash

app.mount("/", WSGIMiddleware(app_dash.server))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8050, reload=True)
