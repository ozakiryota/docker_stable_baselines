#!/bin/bash

image="stable_baselines"
tag="latest"
exec_pwd=$(cd $(dirname $0); pwd)

docker build $exec_pwd \
    -t $image:$tag \
    --build-arg cache_bust=$(date +%s)