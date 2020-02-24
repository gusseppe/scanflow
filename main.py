import os
import pandas as pd
from autodeploy.check import Checker
from pandas_profiling import ProfileReport

app_dir = '/home/guess/Desktop/autodeploy/examples/demo2/data-science/'
checker = Checker(app_dir)

df_track = checker.get_tracked_values(workflow_name='workflow1',
                                      executor_name='preprocessing')
print(df_track.shape)
cols = ['params.n_classes', 'params.n_samples']
df_track[cols].head()

profile = ProfileReport(df_track)
profile
