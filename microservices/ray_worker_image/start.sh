#!/bin/bash

echo "$RAYHEAD"

ray start --address="$RAYHEAD":6379 --object-manager-port=12345 --node-manager-port=12346

tail -f /dev/null