#!/bin/bash

# more gnn layer with esm feat and skip-connection
splits=("a" "b" "c")
neg_sampling_ratios=(1 5 10 20)

for _split in "${splits[@]}"; do
    for _ratio in "${neg_sampling_ratios[@]}"; do
        python train_batch_pl.py --device 2 \
        --beta 0.4 --split $_split  --in_channels 1280 \
        --exp_name neg_sampling_ratio_exp \
        --hidden_channels 512 --disjoint_train_ratio 0.3 \
        --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
        --pep_feat esm  --epoch 20 --neg_sampling_ratio $_ratio --gnn_method gat_conv \
        --gnn_layers 3
    done
done