source activate fairseq
module load gcc/9.3.0
cd /exp/xzhang/slt/fairseq/examples/MMPT
#python locallaunch.py projects/mfmmlm/how2sign.yaml
python locallaunch.py projects/mtm/vlm/how2sign.yaml
#python locallaunch.py projects/mtm/vlm/how2sign_test.yaml
