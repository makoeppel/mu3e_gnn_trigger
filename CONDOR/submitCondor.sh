#!/bin/bash

echo "Working dir: $PWD"
source ~/mu3e_trigger/venv/bin/activate

echo "Command: $@"
eval "$@"