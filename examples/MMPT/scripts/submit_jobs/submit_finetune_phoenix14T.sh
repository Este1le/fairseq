qsub -l mem_free=200G,h_rt=200:00:00,gpu=1 -q gpu.q@@RTX /exp/xzhang/slt/fairseq/examples/MMPT/scripts/submit_jobs/finetune_phoenix14T.sh
