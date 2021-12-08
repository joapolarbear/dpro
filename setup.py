#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

name = "dpro"
version = "0.0.1"
description = "A profiling, replay and optimization toolkit for distributed DNN training"

rootdir = os.path.abspath(os.path.dirname(__file__))
long_description = open(os.path.join(rootdir, "README.md")).read()

setup(
    name=name,
    version=version,
    description=description,

    # Long description of your library
    long_description=long_description,
    long_description_content_type = 'text/markdown',

    url='https://github.com/joapolarbear/dpro.git',
    author='Hanpeng Hu',
    author_email='hphu@cs.hku.hk',
    license='MIT',
    packages=find_packages(),

    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
    ],
    # package_dir={"dpro": "dpro"},
    # python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
)