#!/bin/bash

cd /home/ray

ray start --head --port="$HOST_PORT" --object-manager-port="$OBJECT_MANAGER_PORT" --node-manager-port="$NODE_MANAGER_PORT" --node-ip-address="$NODE_IP_ADDRESS" --dashboard-port="$DASHBOARD_PORT" --dashboard-host 0.0.0.0 --num-cpus=0

cd /usr/src/distributed_training

python server.py
