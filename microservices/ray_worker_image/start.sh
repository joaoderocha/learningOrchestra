#!/bin/bash

echo "$RAYHEAD"

ray start --address="$RAYHEAD":6379

tail -f /dev/null