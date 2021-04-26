import uvicorn
import logging
import sys

from scanflow.server import ui
from scanflow.server import api
from fastapi.middleware.wsgi import WSGIMiddleware

import mlflow
from mlflow.tracking import MlflowClient

server_port = sys.argv[1]
mlflow_port = sys.argv[2]
mas_port = sys.argv[3]

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

try:
    tracker_uri = f"http://0.0.0.0:{mlflow_port}"
    # mlflow.set_tracking_uri(tracker_uri)
    client = MlflowClient(tracker_uri)
    experiments = client.list_experiments() # Check if tracker_uri exists
except:
    client = None



# Get FastAPI and Dash app objects
app_fastapi = api.get_app_fastapi(client, mas_port)
app_dash = ui.get_app_dash(client, mlflow_port, server_port)

# Join Dash into FastAPI using a Middleware
app_fastapi.mount("/", WSGIMiddleware(app_dash.server))


if __name__ == "__main__":
    uvicorn.run("server:app_fastapi", host="0.0.0.0",
                port=int(server_port), reload=True)
