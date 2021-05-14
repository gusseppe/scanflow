from mlflow.tracking import MlflowClient
import logging
logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = MlflowClient()

def get_metadata(experiment_name='Default', executor_name=None, metric=None, param=None, client=None):
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"[Agent]  '{experiment_name}' experiment loaded.")
    else:
        logging.info(f"[Agent]  '{experiment_name}' experiment does not exist. Please enter another experiment.")

        return None

    query = f"tag.mlflow.runName='{executor_name}'"
    runs_info = client.search_runs(experiment_id, query,
                                   order_by=["attribute.start_time DESC"],
                                   max_results=1)
    if runs_info:
        run_info = runs_info[0]
        if metric:
            if metric in run_info.data.metrics.keys():
                metric_value = run_info.data.metrics[metric]
                return metric_value
            else:
                print(f'Metric = [{metric}] does not exist.')
                return None
        if param:
            if param in run_info.data.params.keys():
                param_value = run_info.data.params[param]
                return param_value
            else:
                print(f'Param = [{param}] does not exist.')
                return None
        if metric is None and param is None:
            data = run_info.data
            return data

        print(f'Please enter a valid metric or param.')
        return None
    else:
        print(f'RunName=[{executor_name}] does not exist.')
        return None

def get_agents_uri():

    # TODO: get this information from mlflow
    supervisor_ctn = "inference-mnist-supervisor-agent"
    supervisor_port = 8003

    checker_ctn = "inference-mnist-checker-agent"
    checker_port = 8004

    improver_ctn = "inference-mnist-improver-agent"
    improver_port = 8005

    planner_ctn = "inference-mnist-planner-agent"
    planner_port = 8006

    supervisor_agent_uri = f"http://{supervisor_ctn}:{supervisor_port}"
    checker_agent_uri = f"http://{checker_ctn}:{checker_port}"
    improver_agent_uri = f"http://{improver_ctn}:{improver_port}"
    planner_agent_uri = f"http://{planner_ctn}:{planner_port}"

    agents_uri = {
        'supervisor': supervisor_agent_uri,
        'checker': checker_agent_uri,
        'improver': improver_agent_uri,
        'planner': planner_agent_uri,
    }

    return agents_uri

def get_host_uri():

    host_gateway = get_metadata(experiment_name="Scanflow",
                              executor_name='info',
                              param='host_gateway')

    host_port = 8050 # TODO: try to get the port from mlflow

    host_uri = f'http://{host_gateway}:{host_port}'

    return host_uri

def get_app_dir():
    app_dir = get_metadata(experiment_name="Scanflow",
                                executor_name='info',
                                param='app_dir')

    return app_dir
