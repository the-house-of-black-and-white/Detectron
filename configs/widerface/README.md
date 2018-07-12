```bash

xhost + && nvidia-docker run --privileged --rm -it \
-e DISPLAY=unix$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /mnt/hdd/datasets:/ds \
-v /mnt/hdd/datasets/mapping-challenge:/detectron/lib/datasets/data/crowdAIMappingChallenge \
-v ${PWD}:/detectron \
housebw/detectron bash

python2 tools/train_net.py \
    --cfg configs/widerface/retinanet_R-50-FPN_1x.yaml \
    OUTPUT_DIR .tmp/output

python2 tools/train_net.py \
    --cfg configs/crowdAIMappingChallenge/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR .tmp/output DOWNLOAD_CACHE /ds/detectron-zoo

python2 tools/train_net.py \
--cfg configs/crowdAIMappingChallenge/e2e_mask_rcnn_R-50-FPN_1x.yaml \
OUTPUT_DIR .tmp/output \
DOWNLOAD_CACHE /ds/detectron-zoo



json_stats: {"eta": "2 days, 3:34:27", "fl_fpn3": 0.004245, "fl_fpn4": 0.001068, "fl_fpn5": 0.000268, "fl_fpn6": 0.041060, "fl_fpn7": 0.000017, "iter": 0, "loss": 0.447119, "lr": 0.006667, "mb_qsize": 64, "mem": 6309, "retnet_bg_num": 660054.000000, "retnet_fg_num": 55.000000, "retnet_loss_bbox_fpn3": 0.000000, "retnet_loss_bbox_fpn4": 0.000000, "retnet_loss_bbox_fpn5": 0.187777, "retnet_loss_bbox_fpn6": 0.027284, "retnet_loss_bbox_fpn7": 0.185400, "time": 16.503753}
json_stats: {"eta": "7:34:00", "fl_fpn3": 0.002588, "fl_fpn4": 0.000608, "fl_fpn5": 0.000122, "fl_fpn6": 0.000022, "fl_fpn7": 0.000005, "iter": 20, "loss": 0.419723, "lr": 0.007200, "mb_qsize": 64, "mem": 6336, "retnet_bg_num": 659735.000000, "retnet_fg_num": 186.500000, "retnet_loss_bbox_fpn3": 0.164936, "retnet_loss_bbox_fpn4": 0.076835, "retnet_loss_bbox_fpn5": 0.038217, "retnet_loss_bbox_fpn6": 0.000000, "retnet_loss_bbox_fpn7": 0.000000, "time": 2.425659}
json_stats: {"eta": "5:24:11", "fl_fpn3": 0.003548, "fl_fpn4": 0.000888, "fl_fpn5": 0.000183, "fl_fpn6": 0.000047, "fl_fpn7": 0.000009, "iter": 40, "loss": 0.471960, "lr": 0.007733, "mb_qsize": 64, "mem": 6342, "retnet_bg_num": 659984.500000, "retnet_fg_num": 100.500000, "retnet_loss_bbox_fpn3": 0.027224, "retnet_loss_bbox_fpn4": 0.103657, "retnet_loss_bbox_fpn5": 0.091148, "retnet_loss_bbox_fpn6": 0.012547, "retnet_loss_bbox_fpn7": 0.000000, "time": 1.735185}
json_stats: {"eta": "5:22:26", "fl_fpn3": 0.007902, "fl_fpn4": 0.001496, "fl_fpn5": 0.000164, "fl_fpn6": 0.000033, "fl_fpn7": 0.000007, "iter": 60, "loss": 0.423602, "lr": 0.008267, "mb_qsize": 64, "mem": 6345, "retnet_bg_num": 659829.500000, "retnet_fg_num": 155.000000, "retnet_loss_bbox_fpn3": 0.085502, "retnet_loss_bbox_fpn4": 0.097893, "retnet_loss_bbox_fpn5": 0.013434, "retnet_loss_bbox_fpn6": 0.014196, "retnet_loss_bbox_fpn7": 0.000000, "time": 1.728917}
json_stats: {"eta": "5:17:48", "fl_fpn3": 0.005390, "fl_fpn4": 0.030118, "fl_fpn5": 0.000153, "fl_fpn6": 0.000025, "fl_fpn7": 0.000004, "iter": 80, "loss": 0.427864, "lr": 0.008800, "mb_qsize": 64, "mem": 6345, "retnet_bg_num": 659700.000000, "retnet_fg_num": 244.500000, "retnet_loss_bbox_fpn3": 0.120426, "retnet_loss_bbox_fpn4": 0.068709, "retnet_loss_bbox_fpn5": 0.022581, "retnet_loss_bbox_fpn6": 0.002541, "retnet_loss_bbox_fpn7": 0.000000, "time": 1.707114}
json_stats: {"eta": "5:18:19", "fl_fpn3": 0.009565, "fl_fpn4": 0.010054, "fl_fpn5": 0.003752, "fl_fpn6": 0.000063, "fl_fpn7": 0.000007, "iter": 100, "loss": 0.471620, "lr": 0.009333, "mb_qsize": 64, "mem": 6345, "retnet_bg_num": 659723.500000, "retnet_fg_num": 205.000000, "retnet_loss_bbox_fpn3": 0.134051, "retnet_loss_bbox_fpn4": 0.142143, "retnet_loss_bbox_fpn5": 0.055338, "retnet_loss_bbox_fpn6": 0.000000, "retnet_loss_bbox_fpn7": 0.000000, "time": 1.712969}

```