python ./main.py \
    --task test \
    --fname configs/exp_dit_petct/petct_dual.yml \
    --dataset psma \
    --input_mode dual \
    --eval_strategy inversion \
    --eval_step 3 \
    --save_heatmaps 0 \
    --devices cuda:0
