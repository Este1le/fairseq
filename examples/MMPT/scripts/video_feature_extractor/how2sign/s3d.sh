#!/bin/bash

source activate fairseq
cd /exp/xzhang/slt/fairseq/examples/MMPT/
module load gcc/9.3.0
module load ffmpeg/3.3.8
python scripts/video_feature_extractor/extract.py \
    --vdir /exp/xzhang/slt/datasets/how2sign/videos_full/train_raw_videos \
    --fdir /exp/xzhang/slt/datasets/how2sign/features/train_full \
    --type=s3d --num_decoding_thread=4 \
    --batch_size 32 --half_precision 1
