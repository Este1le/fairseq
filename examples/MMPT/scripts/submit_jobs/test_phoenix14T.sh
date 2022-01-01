source activate fairseq
module load gcc/9.3.0
cd /exp/xzhang/slt/fairseq/examples/MMPT
#python locallaunch.py projects/mfmmlm/test_phoenix14T.yaml
python locallaunch.py projects/mtm/vlm/test_phoenix14T.yaml
