qsub -l num_proc=4,mem_free=100G,h_rt=200:00:00,gpu=1 -q gpu.q@@RTX /exp/xzhang/slt/fairseq/examples/MMPT/scripts/video_feature_extractor/phoenix14T/s3d.sh 
