#!/usr/bin/env bash
CONFIG=$1
WORKDIR=$2
NUM_GPUS=$3
for filename in ${WORKDIR}iter_*.pth; do
    echo "config: $CONFIG, filename: $filename"
    /bin/bash tools/dist_test.sh ${CONFIG} ${filename} ${NUM_GPUS} --eval bbox
done