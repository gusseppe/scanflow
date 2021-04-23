import json
import uvicorn
import logging

from scanflow.deploy.deploy import Deploy
from fastapi import FastAPI

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


app = FastAPI(title='Scanflow API',
              description='Server that allows agents to manipulate the user workflow.',
              )

@app.post("/run/workflowname",
          tags=['Running'],
          summary="Run a workflow by name")
async def run_workflow(workflows_metadata_path: str):

    try:
        with open(workflows_metadata_path) as fread:
            setup = json.load(fread)

    except ValueError as e:
        logging.error(f"{e}")
        logging.error(f"[-] Workflows metadata has not yet saved.")

    deployer = Deploy(setup)
    deployer.run_workflows(verbose=True)


    # paths = tools.get_scanflow_paths(content['app_dir'])
    # meta_dir = paths['meta_dir']

    # response = {"feedback": feedback}
    response = ""

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
