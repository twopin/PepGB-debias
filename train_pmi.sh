#!/bin/bash

split_method=$1
cuda_devices=$2

dropout_ratios=(0.3 0.4 0.5)
disjoint_train_ratios=(0.3 0.4 0.5)
gnn_methods=(gat_conv)
feat_srcs=(esm )
hidden_channels=(256 512)
betas=(0.3 0.4 0.5)
dropping_methods=(Dropout DropMessage)

for feat_src in "${feat_srcs[@]}"; do
    for dropout_ratio in "${dropout_ratios[@]}"; do
        for disjoint_train_ratio in "${disjoint_train_ratios[@]}"; do
            for gnn_method  in "${gnn_methods[@]}"; do
                for hidden_channel  in "${hidden_channels[@]}"; do
                    for beta in "${betas[@]}"; do
                        for dropping_method in "${dropping_methods[@]}"; do
                            python train_pmi_split.py \
                            --data_root ./data/pmi_digraph_data \ 
                            --add_skip_connection --add_self_loops \
                            --disjoint_train_ratio 0.4 --batch_size 32 \
                            --dropout_ratio 0.3  --epoch 2
                        done
                    done
                done
            done
        done
    done
done