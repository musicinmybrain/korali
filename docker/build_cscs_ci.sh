#!/bin/bash

set -euo pipefail

ERROR_MSG="Error: enter either 'cpu' or 'gpu' as 1st parameter to choose appropriate base image for target architecture."
if [ "$#" -ne 1 ]; then
    echo "${ERROR_MSG}"
    exit 1
elif [[ "$1" = "cpu" || $1 = "gpu" ]]; then
    echo "Building Docker image with $1 support on base image $(grep FROM Dockerfile_cscs_ci.base.$1 | cut -d ' ' -f2-)."
else
    echo "${ERROR_MSG}"
    exit 1
fi

cd "$(dirname $0)"
COMMIT_SHA=$(git rev-parse HEAD) # for deploy image tag
IMAGE_BASENAME="lukasgd/korali_cscs_ci_$1"

# Setting BUILD_ENV_IMAGE as an environment variable for this script allows building on different base image. In that case you probably als want to provide a custom DEPLOY_IMAGE name
BUILD_ENV_IMAGE=${BUILD_ENV_IMAGE:-"${IMAGE_BASENAME}:base"}
DEPLOY_IMAGE_UNTAGGED=${DEPLOY_IMAGE_UNTAGGED:-"${IMAGE_BASENAME}"}
DEPLOY_IMAGE=${DEPLOY_IMAGE:-"${DEPLOY_IMAGE_UNTAGGED}:${COMMIT_SHA}"}

set -x
if [[ -z $(docker images -q ${BUILD_ENV_IMAGE}) ]]; then
    docker build -f Dockerfile_cscs_ci.base.$1 -t ${BUILD_ENV_IMAGE} .
fi
if [[ -z $(docker images -q ${DEPLOY_IMAGE}) ]]; then
    docker build -f Dockerfile_cscs_ci.deploy.$1 -t ${DEPLOY_IMAGE} .
    if [[ -n ${DEPLOY_IMAGE_UNTAGGED} ]]; then
        docker tag ${DEPLOY_IMAGE} ${DEPLOY_IMAGE_UNTAGGED}
    fi
fi
set +x

