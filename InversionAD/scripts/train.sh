python ./main.py \
    --task train \
    --fname configs/exp_dit_petct/petct_dual.yml \
    --dataset psma \
    --input_mode dual \
    --devices cuda:0 \
    --port 12346
