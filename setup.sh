#!/bin/bash

rm -vr dist/
python setup.py sdist bdist_wheel
pip install -e .
