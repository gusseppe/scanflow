==========
Scanflow [WORK IN PROGRESS]
==========

..
    .. image:: https://img.shields.io/pypi/v/scanflow.svg
        :target: https://pypi.python.org/pypi/scanflow

.. image:: https://readthedocs.org/projects/scanflow/badge/?version=latest
        :target: https://scanflow.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



Machine Learning (ML) is more than just training models, the whole
workflow must be considered. Once deployed, a ML model needs to be watched
and constantly supervised and debugged to guarantee its validity and robustness in
unexpected situations. Debugging in ML aims to identify (and address) the model
weaknesses in not trivial contexts such as bias in classification, model decay,
adversarial attacks, etc., yet there is not a generic framework that allows them to
work in a collaborative, modular, portable, iterative way and, more importantly,
flexible enough to allow both human and machine tasks to work in a multi-agent
environment.

Scanflow allows defining and deploying ML workflows in
containers, tracking their metadata, checking their behavior in production, and
improving the models by using both learned and human-provided knowledge.

Scanflow is a high-level library that is built on top of MLflow and Docker to
manage and supervise workflows efficiently. Its main goals are
usability, integration for deployment and real-time checking.

* Free software: MIT license
* Documentation: https://scanflow.readthedocs.io.


Workflow + Agents
-----------------
.. image:: https://drive.google.com/uc?export=view&id=1lJxQ693Rjr7zYiy2MjDi07dug-uBcGed

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

    import scanflow

    from scanflow.setup import Setup, Executor, Workflow
    from scanflow.special import Tracker, Checker, Improver, Planner
    from scanflow.deploy import Deploy

    # App folder
    base_path = os.path.dirname(os.getcwd())
    app_dir = os.path.join(base_path, "examples/demo_leaf/")

    gathering = Executor(name='gathering',
                           file='gathering.py',
                           dockerfile='Dockerfile_gathering')

    preprocessing = Executor(name='preprocessing',
                           file='preprocessing.py',
                           requirements='req_preprocessing.txt')


    executors = [gathering, preprocessing]

Append your workflows and set a tracker. Besides, you can set
how to run your workflows (sequentially or in parallel).

.. code-block:: python

    # Workflows
    workflow1 = Workflow(name='workflow1',
                     executors=executors,
                     tracker=Tracker(port=8001))


Setup your configuration: build and start the containers. Then,
run each workflow.

.. code-block:: python

    workflows = Setup(app_dir, workflows=[workflow1],
                             verbose=False)

    # Define a deployer to build, start and run the workflows
    deployer = Deploy(app_dir, setup, verbose=False)

    # Build the docker images
    deployer.build_workflows()

    # Start the containers
    deployer.start_workflows()

    # Run the user's code on the containers
    deployer.run_workflows()

All the containers will be shown in the scanflow UI, run the following 
to start the server.

.. code-block:: bash

    python cli.py server --server_port 8050
    
Dashboard alpha
-----------------

- Go to: http://localhost:8050
.. image:: https://drive.google.com/uc?export=view&id=1ii7wyXqsDA-eiyA5pI3Y1yccWg-p4FFC


Tutorials
-----------

Please check the jupyter notebooks for more examples:

.. code-block:: bash

    tutorials/
    
Installation
------------

- Install docker.
- sudo usermod -aG docker <your-user> (on Linux)

**Using conda**

.. code-block:: bash

    conda create -n scanflow python=3.6
    source activate scanflow
    git clone https://github.com/gusseppe/scanflow
    cd scanflow
    pip install -r requirements.txt


**Using pip (not yet available)**

.. code-block:: bash

    pip install scanflow
    
References
------------   

**Colab experiments**
- https://colab.research.google.com/drive/1t0EgpPk5_mEMNb_AvN7yJ_Tz88bqgS9A?usp=sharing
- https://colab.research.google.com/drive/1tdGqg5WAGGhdZEd0HI5ZYQtRFAkrOvJO?usp=sharing
- https://colab.research.google.com/drive/147NDiwsiY86xpUIsM-btMPMSzTktsNd6?usp=sharing
