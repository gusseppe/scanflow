from autodeploy.setup import setup
from autodeploy.run import run

# App folder
app_dir = '/home/guess/Desktop/autodeploy/examples/demo2/data-science/'

# Workflows

# This workflow1 will gather and preprocess the raw data set. Order matters.
workflow1 = [
    {'name': 'gathering',
     'file': 'gathering.py',
     'parameters': {'raw_data_path': 'leaf.csv',
                    'percentage': 1.0},
     'dockerfile': 'Dockerfile_gathering'}, # Provides a dockerfile

    {'name': 'preprocessing',
     'file': 'preprocessing.py',
     'requirements': 'req_preprocessing.txt'}, # Convert requirements.txt to dockerfile

]

# This workflow2 will model preprocess data set.
workflow2 = [
    {'name': 'modeling',
     'file': 'modeling.py',
     'parameters': {'preprocessed_data': 'preprocessed_data.csv',
                    'model_path': 'models',
                    'n_estimators': 10},
     'requirements': 'req_modeling.txt'}, # Convert requirements.txt to dockerfile
    {'name': 'modeling2',
     'file': 'modeling.py',
     'parameters': {'preprocessed_data': 'preprocessed_data.csv',
                    'model_path': 'models',
                    'n_estimators': 20},
     'requirements': 'req_modeling.txt'}, # Convert requirements.txt to dockerfile

]

workflows = [
    {'name': 'workflow1', 'workflow': workflow1,
     'tracker': {'port': 8001}},
    {'name': 'workflow2', 'workflow': workflow2,
     'tracker': {'port': 8002}, 'parallel': True}

]

workflow_datascience = setup.Setup(app_dir, workflows, verbose=True)

# This will build and start the environments
workflow_datascience.run_pipeline()

# Read the platform and workflows
runner = run.Run(workflow_datascience, verbose=True)

# Run the workflow
runner.run_workflows()
