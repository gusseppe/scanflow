import pandas as pd
import mlflow
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.stats import ks_2samp
from datetime import datetime


def add_noise(df, mu=0, sigma=0.1):
    noise = np.random.normal(mu, sigma, [len(df),len(df.columns)])
    df_noise = df + noise
    return df_noise


def join_predictions(input_pred_files):
    df_list = list()
    for file in input_pred_files:
        df_list.append(pd.read_csv(file))

    df_total = pd.concat(df_list, ignore_index=True)

    return df_total


def format_predictions(input_pred_files):
    df_list = list()
    for file in input_pred_files:
        date_file = os.path.basename(file).replace('input_predictions_', '').replace('.csv', '')
        date_file = datetime.strptime(date_file, '%Y_%m_%d_%H_%M_%S')
        data = pd.read_csv(file)
        df_dict = {'date': date_file,
                   'path': file,
                   'data': data}
        df_list.append(df_dict)

    return df_list


def get_input_predictions(checker_dir, periods=1):
    # self.predict()
    try:
        input_pred_files = sorted(os.listdir(checker_dir),
                                  key=lambda x: os.stat(os.path.join(checker_dir, x)).st_mtime,
                                  reverse=True)
        input_pred_files = [os.path.join(checker_dir, file) for file in input_pred_files]

        if len(input_pred_files) != 0:
            if periods > 0:
                return format_predictions(input_pred_files[:periods])
            elif periods == -1:
                return format_predictions(input_pred_files)
            else:
                return None
        else:
            return None

    except OSError as e:
        logging.error(f"{e}")
        logging.error(f"Path does not exist.", exc_info=True)


def search_by_key(key, var):
    if hasattr(var,'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in search_by_key(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in search_by_key(key, d):
                        yield result


