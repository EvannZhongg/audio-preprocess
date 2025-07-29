version=$(date "+%Y%m%d%H%M%S")
version=audio_preprocess_${version}
echo ${version}

#docker login --username=leolxliu https://csighub.tencentyun.com --password=leolxliu

docker build --network=host -f dockerfile -t csighub.tencentyun.com/avchat/audio_preprocess:${version} ./
docker push csighub.tencentyun.com/avchat/audio_preprocess:${version}