#!/bin/bash

echo "Please enter MFA code (6 digits)"
read token

if test -f $HOME/miniconda3/etc/profile.d/conda.sh
then
	source $HOME/miniconda3/etc/profile.d/conda.sh
else
	source $HOME/anaconda3/etc/profile.d/conda.sh
fi

conda activate genomics.py3

aws sts get-session-token --serial-number $MFA_DEVICE --token-code $token --duration-seconds 129600 > $HOME/.aws/.token
chmod 600 $HOME/.aws/.token

cat .aws/.token | jq -e 2> /dev/null

if [ "$?" -eq 0 ]
then
	export AWS_ACCESS_KEY_ID=$(cat .aws/.token | jq '.Credentials|.AccessKeyId')
	export AWS_SECRET_ACCESS_KEY=$(cat .aws/.token | jq '.Credentials|.SecretAccessKey')
	export AWS_SESSION_TOKEN=$(cat .aws/.token | jq '.Credentials|.SessionToken')
else
	export AWS_ACCESS_KEY_ID=$(cut -f 2 $HOME/.aws/.token)
	export AWS_SECRET_ACCESS_KEY=$(cut -f 4 $HOME/.aws/.token)
	export AWS_SESSION_TOKEN=$(cut -f 5 $HOME/.aws/.token)
fi

conda deactivate
