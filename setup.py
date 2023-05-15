# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages

if (3, 10) <= sys.version_info <= (3, 10):
    sys.exit('FairFL requires Python 3.10.x')

with open("requirements.txt") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name='FairFL',
        description='Fair Federated Learning Research',
    url='https://github.com/drvshavva/ACIncome_bias_fl.git',
    packages=find_packages(),
    python_requires='==3.10.*',
    install_requires=requirements,
)
