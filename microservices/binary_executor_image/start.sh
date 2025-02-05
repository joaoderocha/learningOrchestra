#!/bin/bash

# turn on bash's job control
set -m

# Start the primary process and put it in the background
ulimit -n 65536; ray start --head --port=6379 --dashboard-host=0.0.0.0 --no-monitor &

sleep 10 &
# Start the helper process
python server.py

# the my_helper_process might need to know how to wait on the
# primary process to start before it does its work and returns