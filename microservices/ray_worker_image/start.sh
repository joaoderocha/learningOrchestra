#!/bin/bash

echo "$RAYHEAD"

ray start --address="$RAYHEAD":6379 --object-manager-port="$OBJECT_MANAGER_PORT" --node-manager-port="$NODE_MANAGER_PORT" --block
