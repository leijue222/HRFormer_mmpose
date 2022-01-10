#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

# python -m pip install timm
# python -m pip install einops
# python -m pip install mmcv-full==1.2.2
# python -m pip install numpy --upgrade
# python -m pip install -r requirements.txt
# pip install -e .

DATA_ROOT="../MutiTransPose/data/crowdpose"
# DATA_ROOT="/media/yiwei/yiwei-01/datasets/pose/crowdpose"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=12345 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work-dir "$(dirname $0)/../mmpose-logs" \
    --cfg-options data_cfg.bbox_file="$DATA_ROOT/json/crowdpose_test.json" \
                    data.train.ann_file="$DATA_ROOT/json/crowdpose_trainval.json" \
                    data.train.img_prefix="$DATA_ROOT/images/" \
                    data.val.ann_file="$DATA_ROOT/json/crowdpose_test.json" \
                    data.val.img_prefix="$DATA_ROOT/images/" \
                    data.test.ann_file="$DATA_ROOT/json/crowdpose_test.json" \
                    data.test.img_prefix="$DATA_ROOT/images/"
