#!/bin/bash

datadir=/home/bliu/SRC/workspace/pytf/Speaker/data/vox1/
expdir=exp
stage=2

loss_type="class_softmax"
margin_type="Softmax"
att_type="multi_attention"
model_type="cnn_Res50_IR"
segment_type="all"
speaker_num=64
utter_num=8
resume="none"

segment_shift_rate=0.50
min_segment_length=140
max_segment_length=180
min_num_segment=5
max_num_segment=25

lr=0.002
optim_type="adam"
batch_size=128
dist_url="tcp://127.0.0.1:1550"

. ./utils/parse_options.sh

if [ $stage -le 2 ]; then
  # Multiple GPU
  export NGPUS=2
  CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train_speaker.py --dist-url ${dist_url} --cuda --batch_size ${batch_size} --dataroot $datadir --loss_type ${loss_type} --margin_type ${margin_type} --att_type $att_type --model_type $model_type --segment_type $segment_type --speaker_num $speaker_num --utter_num ${utter_num} --resume ${resume} --lr ${lr} --optim_type ${optim_type} --segment_shift_rate $segment_shift_rate --min_segment_length $min_segment_length --max_segment_length $max_segment_length --min_num_segment $min_num_segment --max_num_segment $max_num_segment
  
  #speaker_seq_"$seq_training"_"$model_type"_"$train_type"_"$segment_type"
  # Single GPU
  #CUDA_VISIBLE_DEVICES=3 python3 local/train_speaker.py --cuda --dataroot $datadir --seq_training $seq_training --train_type $train_type --model_type $model_type --segment_type $segment_type --speaker_num $speaker_num --model_name speaker_DSAE_LSTM_MultiAttention_fromRandomInit_All --lr 0.0001 --segment_shift_rate $segment_shift_rate --min_segment_length $min_segment_length --max_segment_length $max_segment_length --min_num_segment $min_num_segment --max_num_segment $max_num_segment --cmvn_file cmvn.npy --resume /home1/cyr/nworks/speaker/train_speaker/exp/speaker_DSAE_LSTM_MultiAttention_fromRandomInit_All/model_best.pth --vad
fi

exit 0;
