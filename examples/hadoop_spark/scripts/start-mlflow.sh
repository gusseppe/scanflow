#!/bin/bash

#exec mlflow ui -h 0.0.0.0 -p 8001 &> /dev/null &
exec mlflow server \
    --backend-store-uri ./mlruns \
    --host 0.0.0.0 -p 8001



