#!/bin/bash

echo "Working dir: $PWD"
source /data/dust/group/atlas/ttreco/venv/bin/activate

echo "Command: $@"
eval "$@"