#!/bin/bash

echo "Please enter MFA code (6 digits)"
read token
conda activate genomics.py3

aws sts get-session-token --serial-number $MFA_DEVICE --token-code $token --duration-seconds 129600 > $HOME/.aws/.token
chmod 600 $HOME/.aws/.token

export AWS_ACCESS_KEY_ID=$(cut -f 2 $HOME/.aws/.token)
export AWS_SECRET_ACCESS_KEY=$(cut -f 4 $HOME/.aws/.token)
export AWS_SESSION_TOKEN=$(cut -f 5 $HOME/.aws/.token)

conda deactivate