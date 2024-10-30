#!/bin/bash

# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gin_conv 

# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gin_conv \
#  --add_skip_connection

# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gin_conv \
#  --random_feat

# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gin_conv \
#  --add_skip_connection --random_feat


# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gat_conv

# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gat_conv \
#  --add_skip_connection

# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gat_conv \
#  --random_feat

# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gat_conv \
#  --add_skip_connection --random_feat

# more gnn layer with esm feat and skip-connection
# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gat_conv \
#  --add_skip_connection --gnn_layers 3

# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gat_conv \
#  --add_skip_connection --gnn_layers 4

# # random feat with more layers and no skip connection
# python train_batch_pl.py --device 2 \
#  --beta 0.4 --split a  --in_channels 1280 \
#  --hidden_channels 512 --disjoint_train_ratio 0.4 \
#  --dropout_ratio 0.5 --feat_src esm --dropping_method DropMessage \
#  --pep_feat esm  --epoch 20 --gnn_method gat_conv \
#  --random_feat --gnn_layers 3

python train_batch_pl_cmp.py --device 2 \
 --beta 0.4 --split a  --in_channels 1280 \
 --hidden_channels 512 --disjoint_train_ratio 0.3 \
 --data_root_dir "./data_transfer/yipin_protein_peptide/" \
 --dropout_ratio 0.3 --feat_src esm --dropping_method DropMessage \
 --exp_name perturb --perturb_ratio 5 --perturb_method add \
 --pep_feat esm --add_skip_connection  --epoch 20 --gnn_method gat_conv \
 --random_feat --gnn_layers 2