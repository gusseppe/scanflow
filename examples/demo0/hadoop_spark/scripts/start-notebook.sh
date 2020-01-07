#!/bin/bash

exec jupyter lab --ip=0.0.0.0 --NotebookApp.token='' \
            --NotebookApp.password='' \
            --notebook-dir='/root/project' \
            --port=8000 --no-browser \
            --allow-root &> /dev/null &



