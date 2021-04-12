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


@cli.command()
@click.option("--port", "-p", default=8050,
              help="The port to listen on (default: 8050).")
def server(port):

    try:
        print("Starting server")
        command = f"python server/server.py"
        # command = f"python "
        logs_start_server = subprocess.check_output(command.split())
        print(logs_start_server)
    except:
        print('exit')
        sys.exit(1)


if __name__ == '__main__':
    cli()
