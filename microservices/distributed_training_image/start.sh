#!/bin/bash

cd /home/ray

ray start --head --port=6379 --dashboard-port=8265

cd /usr/src/distributed_training

python server.py