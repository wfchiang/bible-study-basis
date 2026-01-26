#!/bin/bash

if [[ "$1" != "ui" && "$1" != "mcp" ]]; then
  echo "Error: Argument must be either 'ui' or 'mcp'"
  echo "Usage: $0 <ui|mcp>"
  exit 1
fi

REGION="us-east4"
PROJECT_ID="weifan-484118"
REPO_NAME="bible-study-basis"
IMAGE_NAME="$1"
TAG="latest"

LOCAL_IMAGE_NAME="${REPO_NAME}-${IMAGE_NAME}"
LOCAL_TAG="latest"

docker tag ${LOCAL_IMAGE_NAME}:${LOCAL_TAG} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}