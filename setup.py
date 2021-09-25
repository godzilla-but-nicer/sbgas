#!/usr/bin/env python
# encoding: utf-8

from setuptools import find_packages, setup

setup(
    name="sbgas",
    version="0.1",
    description="seed bank microbial genetic algorithm",
    author="pat",
    packages=find_packages(exclude=("tests",)),
)
