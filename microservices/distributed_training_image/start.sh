#!/bin/bash

cd /home/ray

ray start --head --port=6379 --object-manager-port=12345 --node-manager-port=12346 --dashboard-port=8265

cd /usr/src/distributed_training

python server.py