#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
CONFIG=$4
WORK_DIR=$5
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
MEM=${MEM:-80G}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:6}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:a100:${GPUS_PER_NODE} \
    --nodes=1 \
    --mem=${MEM} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
