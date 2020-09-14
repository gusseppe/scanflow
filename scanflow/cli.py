import json
import shlex
import os
import sys
import logging
from mlflow.utils import  process
from mlflow.utils.process import ShellCommandException

import click
from click import UsageError
from scanflow.ui import _build_gunicorn_command

@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
# @click.option("--backend-store-uri", metavar="PATH",
#               default=DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
#               help="URI to which to persist experiment and run data. Acceptable URIs are "
#                    "SQLAlchemy-compatible database connection strings "
#                    "(e.g. 'sqlite:///path/to/file.db') or local filesystem URIs "
#                    "(e.g. 'file:///absolute/path/to/directory'). By default, data will be logged "
#                    "to the ./mlruns directory.")
# @click.option("--default-artifact-root", metavar="URI", default=None,
#               help="Path to local directory to store artifacts, for new experiments. "
#                    "Note that this flag does not impact already-created experiments. "
#                    "Default: " + DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH)
@click.option("--port", "-p", default=8050,
              help="The port to listen on (default: 8050).")
def ui(port):
    """
    Launch the MLflow tracking UI for local viewing of run results. To launch a production
    server, use the "mlflow server" command instead.
    The UI will be visible at ``http://localhost:5000`` by default.
    """

    # Ensure that both backend_store_uri and default_artifact_uri are set correctly.

    # TODO: We eventually want to disable the write path in this version of the server.
    try:
        # _run_server(backend_store_uri, default_artifact_root, "127.0.0.1", port, None, 1)
        workers = 2
        host = "127.0.0.1"
        full_command = _build_gunicorn_command(None, host, port, workers or 4)
        # cmd = 'python ui/index.py'
        process.exec_cmd(full_command,  stream_output=True)

    except ShellCommandException:
        # eprint("Running the mlflow server failed. Please see the logs above for details.")
        print('exit')
        sys.exit(1)

def _validate_static_prefix(ctx, param, value):  # pylint: disable=unused-argument
    """
    Validate that the static_prefix option starts with a "/" and does not end in a "/".
    Conforms to the callback interface of click documented at
    http://click.pocoo.org/5/options/#callbacks-for-validation.
    """
    if value is not None:
        if not value.startswith("/"):
            raise UsageError("--static-prefix must begin with a '/'.")
        if value.endswith("/"):
            raise UsageError("--static-prefix should not end with a '/'.")
    return value

if __name__ == '__main__':
    cli()
