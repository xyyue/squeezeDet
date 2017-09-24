#!/bin/bash
for i in {2000..10000..500}
do
    echo $i

	python ./src/eval.py \
	--dataset=KITTI \
	--data_path=./data/KITTI \
	--image_set=val \
	--eval_dir=/tmp/bichen/logs/SqueezeDet/eval_val \
	--checkpoint_path=/home/xyyue/ckpts_half_synth/model.ckpt-$i \
	--net=squeezeDet \
	--gpu=0
done
