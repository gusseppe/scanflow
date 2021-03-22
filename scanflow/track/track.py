# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import os

import docker
import logging
import mlflow

from pathlib import Path
from textwrap import dedent
from scanflow.track.tools import get_input_predictions, search_by_key
from scanflow import tools

logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

client = docker.from_env()


class Tracker:
    def __init__(self, app_dir, verbose=True):
        """
        Example:
            deploy = Track(api_name, port)

        Parameters:
            app_dir (str): Path to the application.

        """
        # self.api_name = api_container_name
        self.app_dir = app_dir
        self.ad_paths = tools.get_scanflow_paths(app_dir)
        self.verbose = verbose
        tools.check_verbosity(verbose)
        self.workflows = tools.read_workflows(self.ad_paths)
        # self.ad_stuff_dir = os.path.join(app_dir, 'ad-stuff')
        # self.ad_meta_dir = os.path.join(self.ad_stuff_dir, 'ad-meta')
        # self.ad_tracker_dir = os.path.join(self.ad_stuff_dir, 'ad-tracker')
        # self.ad_checker_dir = os.path.join(self.ad_stuff_dir, 'ad-checker')
        # self.ad_checker_pred_dir = os.path.join(self.ad_checker_dir, 'predictions')
        # self.ad_checker_model_dir = os.path.join(self.ad_checker_dir, 'model')
        # self.ad_checker_scaler_dir = os.path.join(self.ad_checker_dir, 'scaler')

    def pipeline(self):
        # self.predict()

        return self

    def get_tracker(self, workflow_name='workflow1'):

        tracker_uri = self.get_tracker_uri(workflow_name)

        tracker = mlflow.tracking.MlflowClient(tracking_uri=tracker_uri)

        if tracker_uri is not None:
            return tracker
        else:
            return None

    def get_tracker_dir(self, workflow_name='workflow1'):
        tracker_dirs = [next(search_by_key('tracker_dir', e)) for e in self.workflows]
        tracker_wflow_dir = [tdir for tdir in tracker_dirs if workflow_name in tdir]


        if len(tracker_wflow_dir) == 0:
            return None
        else:
            return tracker_wflow_dir[0]

    def get_tracker_url(self, workflow_name='workflow1'):
        workflows = [wflow for wflow in self.workflows if wflow['name'] == workflow_name]
        tracker_url = [next(search_by_key('url', e)) for e in workflows]

        if len(tracker_url) == 0:
            return None
        else:
            return tracker_url[0]

    def open_browser(self, workflow_name='workflow1'):
        try:
            import webbrowser
            url = self.get_tracker_url(workflow_name)
            webbrowser.open(url)
        except Exception as e:
            print(e)
            print('Error while opening browser.')

    def get_tracker_uri(self, workflow_name='workflow1'):
        tracker_dir = self.get_tracker_dir(workflow_name)

        if tracker_dir is not None:
            tracker_uri = f'file://{tracker_dir}/mlruns/'
            return tracker_uri
        else:
            return None

    # def get_artifacts(self, workflow_name, experiment_name='Default', artifact_name='input.npy'):
    #     tracker_port = '8002'
    #     tracker_uri = f"http://localhost:{tracker_port}"
    #
    #     tracker = mlflow.tracking.MlflowClient(tracking_uri=tracker_uri)
    #
    #     experiment = tracker.get_experiment_by_name(experiment_name)
    #     experiment_id = experiment.experiment_id
    #
    #     runs_info = tracker.list_run_infos(experiment_id,
    #                                       order_by=["attribute.start_time DESC"])
    #     if runs_info:
    #         last_run_id = runs_info[0]
    #         tracker_dir = self.get_tracker_dir(workflow_name)
    #         artifact_path = os.path.join(tracker_dir, last_run_id.artifact_uri[1:], artifact_name)
    #
    #         return artifact_path
    #     else:
    #         return None
    #
    #
    #     return artifacts

    def list_artifacts(self, workflow_name, run_id=None, experiment_name='Default'):
        tracker_port = '8002'
        tracker_uri = f"http://localhost:{tracker_port}"

        tracker = mlflow.tracking.MlflowClient(tracking_uri=tracker_uri)
        tracker_dir = self.get_tracker_dir(workflow_name)


        if run_id is None: # Return the last experiment run
            # exp_dir = os.path.join(tracker_dir, f"mlruns/{experiment_id}")
            # paths = sorted(Path(exp_dir).iterdir(), key=os.path.getmtime, reverse=True)
            # run_id = os.path.basename(str(paths[0]))

            logging.info(f"[Tracker]  'run_id' is not provided. Loading the latest experiment.")
            experiment = tracker.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                if experiment_name == 'Default':
                    logging.info(f"[Tracker]  'experiment_name' is not provided. Loading the 'Default' experiment.")
            else:
                logging.info(f"[Tracker]  Please provide a valid experiment name.")

                return None

            runs_info = tracker.list_run_infos(experiment_id,
                                              order_by=["attribute.start_time DESC"])
            run_id = runs_info[0].run_id


        run = tracker.get_run(run_id)

        artifact_dir = run.info.artifact_uri.replace('/mlflow', tracker_dir)

        artifacts = {artifact : os.path.join(artifact_dir, artifact)
                     for artifact in os.listdir(artifact_dir)}

        return artifacts

    def get_tracked_values(self, workflow_name='workflow1',
                           executor_name=None, verbose=False, **args):
        """
            This function will provide all the tracked values for each workflow.

            For search syntax see: https://www.mlflow.org/docs/latest/search-syntax.html
        """
        tracker_dir = self.get_tracker_dir(workflow_name)
        if tracker_dir is not None:
            # tracker_uri = f'file://{tracker_dir}/mlruns/'
            tracker_uri = self.get_tracker_uri(workflow_name)
            mlflow.set_tracking_uri(tracker_uri)
            df = mlflow.search_runs(['0'], **args) # By now use this because of dataframe output
            col_executor_name = 'tags.mlflow.runName'
            if verbose:
                try:
                    if executor_name is not None:

                        return df[df[col_executor_name] == executor_name]
                    else:
                        return df
                except KeyError as e:
                    logging.error(f"{e}")
                    logging.warning(f"There is no executor with name: {executor_name}.")
            else:
                not_cols = ['experiment_id', 'status',
                            'artifact_uri', 'tags.mlflow.source.type',
                            'tags.mlflow.user']

                try:
                    if executor_name is not None:
                        return df[df.columns.difference(not_cols)][df[col_executor_name] == executor_name]
                    else:
                        return df[df.columns.difference(not_cols)]
                except KeyError as e:
                    logging.error(f"{e}")
                    logging.warning(f"There is no executor with name: {executor_name}.")
        else:
            logging.warning(f"There is no metadata for {workflow_name}.")

            return None


    def __repr__(self):
        workflows_names = [d['name'] for d in self.workflows]
        _repr = dedent(f"""
        Tracker = (
            Workflows: {workflows_names}
            App directory: {self.app_dir}
        )
        """)
        return _repr
