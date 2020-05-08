from autodeploy.check import Checker
from autodeploy.track import Tracker


app_dir = '/home/guess/Desktop/autodeploy/examples/demo2/data-science/'
tracker = Tracker(app_dir)
df = tracker.get_tracked_values(workflow_name='workflow1',
                                executor_name='preprocessing')

checker = Checker(tracker)
checker.explore(df[['params.n_classes', 'params.n_samples']])
