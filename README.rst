==========
Scanflow
==========


.. image:: https://img.shields.io/pypi/v/autodeploy.svg
        :target: https://pypi.python.org/pypi/autodeploy

.. image:: https://readthedocs.org/projects/autodeploy/badge/?version=latest
        :target: https://autodeploy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://github.com/gusseppe/scanflow/blob/master/pictures/autodeploy.png


Scalable library for end-to-end Machine Learning workflow management.

Scanflow is a high-level library that is built on top of MLflow and Docker to
manage and supervise workflows efficiently. Its main goals are
usability, integration for deployment and real-time checking.

* Free software: MIT license
* Documentation: https://autodeploy.readthedocs.io.


Features
--------

- Ease of use, fast prototyping. Write once use everywhere.
- Portability, scalability (based on docker containers).
- Dynamic nested and parallel workflow execution.
- Workflow tracking (e.g, logs, metrics, settings, results, etc.).
- Workflow checking (e.g, drift distribution, quality data, etc.).
- Orchestrator-agnostic.
- Model version control.

Getting started
---------------

Define your working folder and workflows.

.. code-block:: python

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
                        'model_path': 'models2',
                        'n_estimators': 20},
         'requirements': 'req_modeling.txt'},

    ]

Append your workflows and set a tracker. Besides, you can set
how to run your workflows (sequentially or in parallel).

.. code-block:: python

    # Workflows
    workflows = [
        {'name': 'workflow1', 'workflow': workflow1,
         'tracker': {'port': 8001}},
        {'name': 'workflow2', 'workflow': workflow2,
         'tracker': {'port': 8002}, 'parallel': True}

    ]


Setup your configuration: build and start the containers. Then,
run each workflow.

.. code-block:: python

    workflow_datascience = setup.Setup(app_dir, workflows, verbose=False)

    # This will build and start the environments
    workflow_datascience.run_pipeline()

    # Read the platform and workflows
    runner = run.Run(workflow_datascience, verbose=True)

    # Run the workflow
    runner.run_workflows()


Installation
------------

- Install docker.

.. code-block:: bash

    pip install scanflow (not yet)
