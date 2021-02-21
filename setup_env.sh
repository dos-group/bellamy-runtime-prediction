#!/bin/bash

conda env create -f environment.yml

source activate bellamyV1

# causes issues with ray, but is apparently weird dependency of torch
pip uninstall -y dataclasses
