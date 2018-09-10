#!/bin/bash

gpu=$1
shift

session=$1
shift

DIR=/project

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$gpu" ${cmd} run -d \
        --net host \
        --name "$session" \
        -v `pwd`/:$DIR:rw \
        -t max/pomrl \
        $@

