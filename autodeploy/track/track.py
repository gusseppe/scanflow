# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

import json
import time

import docker
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

client = docker.from_env()


class Track:
    def __init__(self,
                 api_name='app_single_api',
                 port=5001):
        """
        Example:
            deploy = Track(api_name, port)

        Parameters:
            api_name (str): API's container name.
            port (int): API's port.

        """
        self.api_name = api_name
        self.port = port

    def pipeline(self):
        self.predict()

        return self

    def predict(self, sample_input):
        """
        Use the API to predict with a given input .

        Parameters:
            sample_input (str): Input sample.
        Returns:
            response_json (dict): prediction.
        """
        url = f'http://localhost:{self.port}/invocations'

        start = time.time()
#         for _ in range(n_requests):
        response = requests.post(
                     url=url, data=json.dumps(sample_input),
                      headers={"Content-type": "application/json; format=pandas-split"})
        response_json = json.loads(response.text)
            #print(response_json)

        end = time.time()
        print(f'Container in port: {self.port}')
        print(f'Time elapsed: {end-start}')

        return response_json


def test_corr(x1, x2):
    """
      Test correlation matrix similarity using
      signed distances.

      Parameters
      ----------
      x1 : DataFrame.
          Numerical features of the original data.
      x2 : DataFrame.
          Numerical features of the new data.

      Returns
      -------
      corr_df : DataFrame
          Dataframe with the distances.
      explain_df : DataFrame
          Dataframe with the explanations.
    """

    def signed_distance(x1, x2):
        df_t = (x1.corr().abs() - x2.corr().abs()).abs()
        df_t = df_t.where((x1.corr() * x2.corr()) > 0, -df_t)

        return df_t

    def explain_distance(df):
        conditions = [
            (df > 0) & (df <= 0.1),
            (df > 0.1) & (df <= 0.2),
            (df > 0.2),
            (df < 0)]

        choices = ['good', 'medium', 'bad', 'diff direction']
        df = np.select(conditions, choices, default='diagonal')

        return pd.DataFrame(df)

    corr_df = signed_distance(x1, x2)
    explain_df = explain_distance(corr_df)

    return corr_df, explain_df


def test_ks(x1, x2):
    """
      Test Kolmogorov-Smirnov.
      The null hypothesis (H0) is that these two variables
      are drawn from same continuous distribution.

      Parameters
      ----------
      x1 : DataFrame.
          Numerical features of the original data.
      x2 : DataFrame.
          Numerical features of the new data.

      Returns
      -------
      ks_df : DataFrame
          Dataframe with the p-values and explanations.
    """

    def explainer_ks(p_value):
        explain = ''
        # interpretation
        alpha = 0.1
        if p_value < alpha:
            explain = 'Different'  # reject H0
        elif (p_value >= alpha) and (p_value <= 0.4):
            explain = 'Slightly different'  # fail to reject H0
        else:
            explain = 'Comparable'  # fail to reject H0
        return explain

    p_value_dict = {}
    for col in x1:
        stat, p_value = ks_2samp(x1[col], x2[col])
        p_value_dict[col] = [p_value, explainer_ks(p_value)]

    ks_df = pd.DataFrame(p_value_dict, index=['p_value', 'explain'])

    return ks_df


def overall_test(x1, x2, test=['ks'], verbose=False):
    print(f'########## Comparing two numerical dataframes ##########')
    if 'ks' in test:
        print()
        print(f'######## Kolmogorov-Smirnov test ########')
        df_ks = test_ks(x1, x2)
        if verbose:
            #       x1.plot(kind='density', title='Old matrix')
            fig = plt.figure(figsize=[7, 7])
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            plt.subplots_adjust(hspace=0.3)
            for col in x1:
                _ = sns.kdeplot(x1[col], ax=ax1).set_title("Old matrix")
            for col in x2:
                _ = sns.kdeplot(x2[col], ax=ax2).set_title("New matrix")
            #       _ = sns.distplot(x1).set_title("Old matrix")
            #       x2.plot(kind='density', title='New matrix')
            plt.show()

        print(df_ks)

    if 'corr' in test:
        print()
        print(f'######## Correlation matrix similarity test ########')
        corr_df, explain_df = test_corr(x1, x2)
        if verbose:
            fig = plt.figure(figsize=[7, 7])
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            plt.subplots_adjust(hspace=0.3)
            _ = sns.heatmap(x1.corr(), vmin=-1, vmax=1,
                            annot=True, ax=ax1).set_title("Old matrix")
            pp1 = sns.pairplot(x1, height=1.5)
            pp1.fig.suptitle("Old matrix", y=1.02)

            _ = sns.heatmap(x2.corr(), vmin=-1, vmax=1,
                            annot=True, ax=ax2).set_title("New matrix")
            pp2 = sns.pairplot(x2, height=1.5)
            pp2.fig.suptitle("New matrix", y=1.02)
            plt.show()
            print(corr_df)

        print(explain_df)
