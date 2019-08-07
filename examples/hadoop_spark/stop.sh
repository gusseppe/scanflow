#!/bin/bash

# Take care of it. It shutdown all the running containers.
docker stop $(docker ps -a -q)
docker container prune
