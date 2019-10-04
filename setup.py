# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-04 15:55:09
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-04 16:08:44
from setuptools import setup

setup(name='sc2_env',
    version='0.0.1',
    install_requires=[
        'gym',
        'pysc2',
        'absl-py',
        'numpy',
        'scipy'
    ]
)
