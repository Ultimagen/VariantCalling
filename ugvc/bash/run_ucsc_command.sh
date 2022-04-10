#!/bin/bash

# runs bedGraphToBigWig through specified environment

set -e

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 cmd args" >&2
  exit 1
fi

conda_env=ucsc
cmd=$1
args=${@:2}
printf "\n"
echo "running $cmd from $(which bedGraphToBigWig)"
orig_conda=$CONDA_DEFAULT_ENV
eval "$(conda shell.bash hook)"
set +e
( conda env list | grep -q $conda_env )
match=$?
set -e
if [ $match -ne 0 ]
  then
    >&2 echo "Please create environment $conda_env from src/setup/other_envs"
    exit 1
fi


conda activate $conda_env
$cmd $args
conda deactivate
conda activate $orig_conda
