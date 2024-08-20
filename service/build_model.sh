#!/usr/bin/env bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

pushd ./ast/
conda env create -f ./environment.yml
conda activate ast
python ./create.py
conda deactivate
popd
