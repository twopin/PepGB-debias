import os
import torch
import torch.nn as nn
import esm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from transformers import AutoConfig, AutoTokenizer
import time
from tqdm import tqdm

class ESM2(nn.Module):
    def __init__(self, dropout, update_layers, model_freeze=True):
        super(ESM2, self).__init__()  
        self.model, _ = esm.pretrained.load_model_and_alphabet("./pretrained_esm/esm2_t33_650M_UR50D.pt")
        self.dropout = nn.Dropout(dropout)

        # freeze parameters
        if model_freeze:
            for name, p in self.model.named_parameters():
                p.requires_grad = False
                for layer_name in update_layers:
                    if name.startswith(layer_name):
                        p.requires_grad = True
        self.output_dim = self.model.embed_dim
    def forward(self, x):
        amino_acid_rep = self.model(x.long(), repr_layers=[33], return_contacts=False)["representations"][33]
        amino_acid_rep = self.dropout(amino_acid_rep)
        return amino_acid_rep

   
class PepCLModel(nn.Module):
    def __init__(self, pretrained_model, pooling, dropout_rate=0.3, 
                 update_layers=['lm_head','layers.30.','layers.31.','layers.32.','emb_layer_norm_after']):
        super(PepCLModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)
        config = AutoConfig.from_pretrained(pretrained_model, force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model,)
        config.attention_probs_dropout_prob = dropout_rate  
        config.hidden_dropout_prob = dropout_rate           
        # self.model = BertModel.from_pretrained(pretrained_model, config=config)
        self.model = ESM2(dropout_rate, update_layers, model_freeze=True)
        self.pooling = pooling
        
    def forward(self, input_ids):

        # out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        out = self.model(input_ids)
        # print(out.shape)

        if self.pooling == 'cls':
            return out[:, 0]  # [batch, 1280]
        
        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 1280]
        
        if self.pooling == 'last-avg':
            last = out.transpose(1, 2)
            # print(torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1).shape)    # [batch, 1280, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 1280]
        
        if self.pooling == 'last-avg-no-transpose':
            last = out
            #print(torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1).shape)    # [batch, 1280, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 1280]


        if self.pooling == 'first-last-avg':
            first = out.transpose(1, 2)    # [batch, 768, 1280]
            last = out.transpose(1, 2)    # [batch, 768, 1280]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 1280]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 1280]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 1280]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 1280]


class PepEsmT12(nn.Module):
    def __init__(self, model):
        super(PepEsmT12, self).__init__()
        self.model = model
        self.output_dim = self.model.out_features

    def forward(self, x):
        amino_acid_rep = self.model(x, repr_layers=[12], return_contacts=False)["representations"][12]
        return amino_acid_rep
    
# class ESM2(nn.Module):
#     def __init__(self, model_path, model_name, model_freeze=False):
#         super(ESM2, self).__init__()

#         model_data = torch.load(os.path.join(model_path,
#                                              "{}.pt".format(model_name)),
#                                 map_location="cpu")
#         regression_data = torch.load(
#             os.path.join(model_path,
#                          '{}-contact-regression.pt'.format(model_name)))
#         self.feature_extractor, self.alphabet = esm.pretrained.load_model_and_alphabet_core(
#             model_name, model_data, regression_data)

#         self.output_dim = self.feature_extractor.embed_dim

#         # freeze parameters
#         if model_freeze:
#             for name, p in self.feature_extractor.named_parameters():
#                 if not name.startswith('lm_head'):
#                     p.requires_grad = False

#     def forward(self, x):

#         amino_acid_rep = self.feature_extractor(
#             x.long(), repr_layers=[12],
#             return_contacts=False)["representations"][12]

#         return amino_acid_rep


class RawESM2(nn.Module):
    def __init__(self, model_name, model_freeze=False):
        super(RawESM2, self).__init__()  
        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        
        # freeze parameters
        if model_freeze:
            for name, p in self.model.named_parameters():
                # if not name.startswith('lm_head'):
                p.requires_grad = False
        # self.output_dim = self.model.embed_dim
        self.output_dim = 1280

    def forward(self, x):
        amino_acid_rep = self.model(x.long(), repr_layers=[33], return_contacts=False)["representations"][33]
        return amino_acid_rep


class ESM_wrap(nn.Module):
    def __init__(self, model):
        super(ESM_wrap, self).__init__()
        self.feature_extractor = model
        self.output_dim = self.feature_extractor.output_dim

    def forward(self, x):
        amino_acid_rep = self.feature_extractor(None, x)['residue_feature']
        return amino_acid_rep


def load_protein_peptide_model(model_name, model_path,model_freeze,type='protein'):
    """ load pre-trained protein/peptide model
    """
    if type=='protein':
        model_path = model_path
        model_protein = RawESM2(model_name, model_freeze=True)

        if model_freeze:
            for param in model_protein.parameters():
                param.requires_grad = False
        print('pre-trained protein model loaded')
        # print(model_protein)
        return model_protein
    else:
        model_peptide = RawESM2(model_name, model_freeze=True)
        if model_freeze:
            for param in model_peptide.parameters():
                param.requires_grad = False
        print('pre-trained peptide model loaded')
        # print(model_protein)
        return model_peptide
    

def split_array(arr, num_splits):
    if num_splits <= 0:
        raise ValueError("num_splits必须大于0")
    
    avg_length = len(arr) // num_splits
    remainder = len(arr) % num_splits
    
    return [arr[i:i+avg_length+(1 if i < remainder else 0)] for i in range(0, len(arr), avg_length+(1 if remainder > 0 else 0))]


def Retrieve_Embedding_From_ESM(sequence_input, file_name, only_token=False):
    model_token_arch = 'ESM-1b'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    tokenizer = Alphabet.from_architecture(model_token_arch).get_batch_converter(800)
    sequence_input = [("protein_{}".format(i), seq) for i, seq in enumerate(sequence_input)]
    _, _, batch_token = tokenizer(sequence_input)

    if only_token:
        print('Loading token...')
        return batch_token

    dataset = AllBatchedDataset(list(range(0, len(batch_token))), batch_token.numpy())
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, num_workers=0, shuffle=False)

    model_name = './pretrained_esm/esm2_t33_650M_UR50D.pt'
    model_protein = RawESM2(model_name, model_freeze=True).to(device)
    model_protein.eval()

    embeddings = []
    with torch.no_grad():
        for batch_idx, (labels, toks) in tqdm(enumerate(data_loader)):
            embedding = model_protein(toks.to(device))
            embeddings.append(embedding.detach().cpu())
            torch.cuda.empty_cache()
        all_embeddings = torch.cat(embeddings, dim=0)

    torch.save(all_embeddings, file_name)
    return all_embeddings


def Retrieve_Embedding_From_PepCL(sequence_input, file_name, pretrained_model_path, only_token=False):
    model_token_arch = 'ESM-1b'

    tokenizer = Alphabet.from_architecture(model_token_arch).get_batch_converter()
    sequence_input = [("peptide_{}".format(i), seq) for i, seq in enumerate(sequence_input)]
    _, _, batch_token = tokenizer(sequence_input)
    if only_token:
        print('Loading token...')
        return batch_token
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    pretrain_path = './pretrained_esm/esm2_t33_650M_UR50D'
    simcsemodel = PepCLModel(pretrain_path, pooling='cls')
    loaded_state_dict = torch.load(pretrained_model_path)
    simcsemodel.load_state_dict(loaded_state_dict)
    model_peptide = simcsemodel.model.to(device)
    model_peptide.eval()

    dataset = AllBatchedDataset(list(range(0, len(batch_token))), batch_token.numpy())
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, num_workers=0, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch_idx, (labels, toks) in enumerate(data_loader):
            embedding = model_peptide(toks.to(device))
            embeddings.append(embedding.detach().cpu())
            torch.cuda.empty_cache()
        all_embeddings = torch.cat(embeddings, dim=0)

    torch.save(all_embeddings, file_name)
    return all_embeddings


class AllBatchedDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]