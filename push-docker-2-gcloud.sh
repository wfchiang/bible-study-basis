#!/bin/bash

REGION="us-east4"
PROJECT_ID="weifan-484118"
REPO_NAME="bible-study-basis"
IMAGE_NAME="app"
TAG="latest"

LOCAL_IMAGE_NAME="${REPO_NAME}-${IMAGE_NAME}"
LOCAL_TAG="latest"

docker tag ${LOCAL_IMAGE_NAME}:${LOCAL_TAG} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}