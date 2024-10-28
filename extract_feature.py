import os
import numpy as np
from typing import List, Dict, Tuple
import argparse
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from transformers import AutoConfig, AutoTokenizer
import pickle
from model import Retrieve_Embedding_From_ESM, Retrieve_Embedding_From_PepCL, SimcseModel


parser = argparse.ArgumentParser(description='Trinity Model')
parser.add_argument("--split_type", type=str, default="random_split",help="Downstream task type")
parser.add_argument('--only_token', default=False, help='using class token')
parser.add_argument("--save_pep_emb_filename", type=str, default='./pep_org.pt', help="peptide embedding save path")
parser.add_argument("--save_prot_emb_filename", type=str, default='./prot_org.pt', help="protein embedding save path")
parser.add_argument("--pep_ckpt", type=str, default='2023_09_05_20_20_05_21500_unsup1.pt', help="peptide embedding ckpt")
parser.add_argument("--pep_fasta", type=str, default='./pep.fasta', help="peptide fasta file")
parser.add_argument("--prot_fasta", type=str, default='./prot.fasta', help="protein fasta file")


def parse_fasta(filename):
    sequences = {}
    current_id = None
    current_sequence = ""
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # Header line
                if current_id is not None:
                    sequences[current_id.split(' ')[0]] = current_sequence
                current_id = line[1:]
                current_sequence = ""
            else:  # Sequence line
                current_sequence += line
    # Add the last sequence
    if current_id is not None:
        sequences[current_id.split(' ')[0]] = current_sequence
    return sequences


prot_fasta_dict = parse_fasta(args.prot_fasta)
protein_sequence = list(fasta_dict.values())

pep_fasta_dict = parse_fasta((args.pep_fasta)
peptide_sequence = list(pep_fasta_dict.values())


print(f"There are {len(protein_sequence)} proteins and {len(peptide_sequence)} peptides.")

protein_emb = Retrieve_Embedding_From_ESM(protein_sequence, args.save_prot_emb_filename, only_token=False)
peptide_emb = Retrieve_Embedding_From_PepCL(peptide_sequence, args.save_prot_emb_filename, './ckpt/'+ args.pep_ckpt, only_token=False)


prot_data = {}
for idx, seq in enumerate(protein_sequence):
    prot_data[seq] = {
        'esm_avg': np.array(torch.mean(protein_emb[idx,:],dim=0)),
        'esm_max': np.array(torch.max(protein_emb[idx,:],dim=0)[0]),
    }

prot_data_len = {}
for idx, seq in enumerate(protein_sequence):
    prot_len = len(seq)
    prot_emb = protein_emb[idx][1:prot_len+1,:]

    prot_data_len[seq] = {
        'esm_avg': np.array(torch.mean(prot_emb,dim=0)),
        'esm_max': np.array(torch.max(prot_emb,dim=0)[0]),
    }

pep_data = {}
for idx, seq in enumerate(peptide_sequence):
    pep_data[seq] = {
        'esm_avg': np.array(torch.mean(peptide_emb[idx,:],dim=0)),
        'esm_max': np.array(torch.max(peptide_emb[idx,:],dim=0)[0]),
    }

pep_data_len = {}
for idx, seq in enumerate(peptide_sequence):
    pep_len = len(seq)
    pep_emb = peptide_emb[idx][1:pep_len+1,:]

    pep_data_len[seq] = {
        'esm_avg': np.array(torch.mean(pep_emb,dim=0)),
        'esm_max': np.array(torch.max(pep_emb,dim=0)[0]),
    }



def extract_feats(src_feat,save_prefix,):
    save_dir = "./"
    mean_feat = {k: v["esm_avg"] for k, v in src_feat.items()}

    # save mean feat
    with open(
        os.path.join(save_dir, "{}_mean.pickle".format(save_prefix)), "wb"
    ) as fout:
        pickle.dump(mean_feat, fout)

save_prefix = "prot_esm"
extract_feats(prot_data,save_prefix)

save_prefix = "pep_esm"
extract_feats(pep_data,save_prefix)

print(f"protein embedding has shape: {protein_emb.shape}.")
print(f"peptide embedding has shape: {peptide_emb.shape}.")