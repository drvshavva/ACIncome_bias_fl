#!/bin/bash

python_base="/home/safir/.virtualenvs/ACIncome_bias_fl/bin/python3"
project_base="/tmp/pycharm_project_99/"

export PYTHONPATH=${project_base}

echo "Starting server"
${python_base} server.py