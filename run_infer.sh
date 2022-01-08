# single-gpu testing

CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/top_down/hrt/coco/hrt_base_coco_256x192.py \
    ./hrt_base_coco_256x192.pth \
    --gpu_collect \
    --out ./hrt_base_coco_256x192_detection.json \
    # [--out ${RESULT_FILE}] [--fuse-conv-bn] \
    # [--eval ${EVAL_METRICS}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--cfg-options ${CFG_OPTIONS}] \
    # [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]
