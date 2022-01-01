qsub -l num_proc=4,mem_free=100G,h_rt=200:00:00,gpu=1 -q gpu.q@@1080 /exp/xzhang/slt/fairseq/examples/MMPT/scripts/video_feature_extractor/how2sign/s3d.sh 
