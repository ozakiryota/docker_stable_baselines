#!/bin/bash

xhost +

image="stable_baselines"
tag="latest"
exec_pwd=$(cd $(dirname $0); pwd)
home_dir="/home/user"

docker run \
	-it \
	--rm \
	-e local_uid=$(id -u $USER) \
	-e local_gid=$(id -g $USER) \
	-e "DISPLAY" \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--gpus all \
	-v $exec_pwd/../pyscr:$home_dir/pyscr \
	-v $exec_pwd/../save:$home_dir/save \
	$image:$tag