#!/bin/bash

set -ex

ECR_REPO="525048827230.dkr.ecr.us-east-1.amazonaws.com"
IMAGE_NAME=ppmseq

sudo -u ${SUDO_USER} aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ECR_REPO}/${IMAGE_NAME}
docker pull ${ECR_REPO}/${IMAGE_NAME}:${TAG}
docker tag ${ECR_REPO}/${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:${TAG}
docker tag ${ECR_REPO}/${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:latest
