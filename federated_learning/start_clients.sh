#!/bin/bash

python_base="/home/safir/.virtualenvs/ACIncome_bias_fl/bin/python3"
project_base="/tmp/pycharm_project_99/"

export PYTHONPATH=${project_base}


for i in `seq 0 5`; do
    echo "Starting client $i"
    ${python_base} client.py $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
# 242792