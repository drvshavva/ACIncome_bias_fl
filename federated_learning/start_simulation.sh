#!/bin/bash

python_base="/home/safir/.virtualenvs/ACIncome_bias_fl/bin/python3"
project_base="/tmp/pycharm_project_99/"

export PYTHONPATH=${project_base}

echo "Starting simulation"
${python_base} server.py

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait