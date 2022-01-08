python demo/top_down_img_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/top_down/hrt/coco/hrt_base_coco_256x192.py \
    ./hrt_base_coco_256x192.pth \
    --img-root demo \
    --img 000000006471.jpg \
    --out-img-root vis_results \
    # [--show --device ${GPU_ID}] \
    # [--kpt-thr ${KPT_SCORE_THR}]klklmkjjkjk