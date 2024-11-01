# Effectively addressing data bias to enhance peptide-protein interaction prediction via graph neural networks

[PepGB & diPepGB](https://arxiv.org/abs/2401.14665) a deep learning-based debiasing framework that mitigates empirical biases and enhances peptide-protein prediction](https://arxiv.org/abs/2401.14665).

<img src="./recomb_figs_1.jpg" alt="model_bias"  width="100%"/>


## Installation


### Install via pip requirments
```bash
pip install requirments.txt
```


## Datasets

Please refer to the `data` folder and download feature pickle files from https://cloud.tsinghua.edu.cn/d/c58334181c494b0eaeba/

## PepGB


### Training PepGB through cross-validation

To train the heterogeneous GNN model PepGB for peptide-protein interaction prediction, please refer to (./ckpt/PepGB/) in the `ckpt` folder. 
Then, run the following command:

```bash
python train_PepGB.py --data_split {split} --outdir ./outputs  # Replace {split} with the setting of the data.
```

### Inference using the trained PepGB ckpt 
To predict your own data, you need to first generate ESM-based features of the peptides and proteins. Please refer to (./feature/README.md) in the `feature` folder. 

Example:

```bash
python inference_pepGB.py 
```

## diPepGB

### Training diPepGB through random-split
```
python train_diPepGB.py
```
For training, please first download the training and validation file  in the `feature` folder.

### inference using the trained diPepGB ckpt
```
python inference_diPepGB.py
```
For inference, using the ckpt model fuile in the `ckpt` folder.

### Downloading ckpts for feature extraction
https://cloud.tsinghua.edu.cn/d/c58334181c494b0eaeba/
