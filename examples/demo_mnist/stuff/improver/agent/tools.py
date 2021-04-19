from mlflow.tracking import MlflowClient
client = MlflowClient()

def get_metadata(executor_name, metric=None, param=None):
    query = f"tag.mlflow.runName='{executor_name}'"
    runs_info = client.search_runs('0', query,
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
        print(f'Please enter a valid metric or param.')
        return None
    else:
        print(f'RunName=[{executor_name}] does not exist.')
        return None
