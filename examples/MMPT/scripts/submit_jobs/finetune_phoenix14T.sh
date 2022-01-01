source activate fairseq
module load gcc/9.3.0
cd /exp/xzhang/slt/fairseq/examples/MMPT
#python locallaunch.py projects/mfmmlm/phoenix14T_de.yaml
python locallaunch.py projects/mtm/vlm/phoenix14T_de.yaml
