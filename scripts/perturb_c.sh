#!/bin/bash

# more gnn layer with esm feat and skip-connection
reduct_ratios=($(seq 5 5 30))
perturbs=("add" "reduct" "mix")

for _perturb in "${perturbs[@]}"; do
    for _ratio in "${reduct_ratios[@]}"; do
        python ../train_batch_pl_cmp.py --device 1 \
        --beta 0.4 --split "c"  --in_channels 1280 \
        --data_root_dir "./data/" \
        --exp_name perturb \
        --hidden_channels 512 --disjoint_train_ratio 0.3 \
        --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
        --perturb_ratio $_ratio --perturb_method $_perturb \
        --pep_feat esm  --epoch 20  --gnn_method gat_conv \
        --gnn_layers 2
    done
done