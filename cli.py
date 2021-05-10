import os
import sys
import subprocess
import logging
# from scanflow.tools import  cmd_start_server

import click

@click.group()
@click.version_option()
def cli():
    pass


@cli.command("server", help="Start the Scanflow UI/API")
@click.option("--server_port", "-p", default=8050,
              help="The port to listen on (default: 8050).")
@click.option("--mlflow_port", default=8002,
              help="The MLflow port (default: 8002).")
@click.option("--mas_port", default=8003,
              help="The MAS (Supervisor agent) port (default: 8003).")
def server(server_port, mlflow_port, mas_port):

    try:
        print("Starting server")
        command = f"python server.py {server_port} {mlflow_port} {mas_port}"
        # command = f"python "
        logs_start_server = subprocess.check_output(command.split())
        print(logs_start_server)
    except:
        print('exit')
        sys.exit(1)


if __name__ == '__main__':
    cli()
