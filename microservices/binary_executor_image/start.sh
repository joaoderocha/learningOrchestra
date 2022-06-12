#!/bin/bash

# turn on bash's job control
set -m

# Start the primary process and put it in the background
ray start --head --dashboard-port=8265 --port=6379 --dashboard-host=0.0.0.0 --redis-password=passwd &

# Start the helper process
python server.py

# the my_helper_process might need to know how to wait on the
# primary process to start before it does its work and returns