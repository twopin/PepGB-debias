#!/bin/bash

# more gnn layer with esm feat and skip-connection
splits=("a" "b" "c")
# from 0 to 0.8,step size 0.01 (changed from 0.05 as requested)
reduct_ratios=($(seq 0 5 30))

for _split in "${splits[@]}"; do
    for _ratio in "${reduct_ratios[@]}"; do
        echo $_ratio
    done
done

# for _split in "${splits[@]}"; do
#     for _ratio in "${reduct_ratios[@]}"; do
#         python train_batch_pl.py --device 1 \
#         --beta 0.4 --split $_split  --in_channels 1280 \
#         --exp_name bias_ratio_exp_0701_pep \
#         --hidden_channels 512 --disjoint_train_ratio 0.3 \
#         --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#         --reduct_ratio $_ratio \
#         --pep_feat esm  --epoch 20  --gnn_method gat_conv \
#         --gnn_layers 3
#     done
# done