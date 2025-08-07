#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd $PROJECT_ROOT
pwd

version=$(date "+%Y%m%d%H%M%S")
version=audio_preprocess_gpu_${version}
echo ${version}

#docker login --username=leolxliu https://csighub.tencentyun.com --password=leolxliu

docker build --network=host -f scripts/dockerfile_gpu -t csighub.tencentyun.com/avchat/audio_preprocess:${version} ./
docker push csighub.tencentyun.com/avchat/audio_preprocess:${version}