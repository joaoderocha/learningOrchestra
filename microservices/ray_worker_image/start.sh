#!/bin/bash

echo "$RAYHEAD"

ray start --address="$RAY_HEAD":"$HOST_PORT" --object-manager-port="$OBJECT_MANAGER_PORT" --node-manager-port="$NODE_MANAGER_PORT"

tail -f /dev/null
