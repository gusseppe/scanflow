
import os
import sys

path = '/home/guess/Desktop/autodeploy'
sys.path.append(path) 

from autodeploy.setup import setup
from autodeploy.run import run

# App folder
app_dir = '/home/guess/Desktop/autodeploy/examples/demo2/data-eng/'

# Workflows
workflow1 = [
    {'name': 'gathering-20200224170602', 'file': 'gathering.py', 
            'env': 'gathering'},

    {'name': 'preprocessing-20200224170602', 'file': 'preprocessing.py', 
            'env': 'preprocessing'}, 

]
workflow2 = [
    {'name': 'modeling-20200224170602', 'file': 'modeling.py', 
            'env': 'modeling'}, 


]
workflows = [
    {'name': 'workflow1', 'workflow': workflow1, 'tracker': {'port': 8001}},
    {'name': 'workflow2', 'workflow': workflow2, 'tracker': {'port': 8002}}

]

workflow_datascience = setup.Setup(app_dir, workflows)


# Read the platform
runner = run.Run(workflow_datascience)

# Run the workflow
runner.run_workflows()

