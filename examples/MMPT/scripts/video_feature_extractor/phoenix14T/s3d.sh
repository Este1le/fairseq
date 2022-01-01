#!/bin/bash
SPLIT0='train' #'dev'
SPLIT1='train' #'val'
source activate fairseq
cd /exp/xzhang/slt/fairseq/examples/MMPT/
module load gcc/9.3.0
module load ffmpeg/3.3.8
python scripts/video_feature_extractor/extract.py \
    --vdir /exp/xzhang/slt/reimp/nslt/Data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/videos/${SPLIT0}/ \
    --fdir /exp/xzhang/slt/fairseq/examples/MMPT/data/phoenix14T/features/${SPLIT1}/ \
    --type=s3d --num_decoding_thread=4 \
    --batch_size 32 --half_precision 1
