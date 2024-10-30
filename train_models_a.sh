#!/bin/bash

cuda_devices=$1

dropout_ratios=(0.3 0.4 0.5)
disjoint_train_ratios=(0.3 0.4 0.5)
gnn_methods=(gat_conv)
feat_srcs=(esm )
hidden_channels=(256 512)
betas=(0.3 0.4 0.5)
dropping_methods=(DropMessage)

# dropout_ratios=(0.5)
# disjoint_train_ratios=(0.4)
# gnn_methods=(gat_conv)
# feat_srcs=(esm)
# hidden_channels=(512)
# betas=(0.4)
# dropping_methods=(DropMessage)

for feat_src in "${feat_srcs[@]}"; do
    for dropout_ratio in "${dropout_ratios[@]}"; do
        for disjoint_train_ratio in "${disjoint_train_ratios[@]}"; do
            for gnn_method  in "${gnn_methods[@]}"; do
                for hidden_channel  in "${hidden_channels[@]}"; do
                    for beta in "${betas[@]}"; do
                        for dropping_method in "${dropping_methods[@]}"; do
                            python train_batch_pl.py  --gnn_method $gnn_method \
                            --disjoint_train_ratio $disjoint_train_ratio \
                            --dropout_ratio $dropout_ratio \
                            --in_channels 1280 \
                            --split_method a \
                            --data_root_dir ./data_transfer/yipin_protein_peptide_a \
                            --neg_sampling_ratio 5.0 \
                            --feat_src $feat_src \
                            --device $cuda_devices \
                            --exp_name exp_1029 \
                            --add_skip_connection \
                            --hidden_channels $hidden_channel \
                            --dropping_method $dropping_method \
                            --beta $beta \
                            --log_root_dir /data/logs/ppi_graph
                        done
                    done
                done
            done
        done
    done
done
