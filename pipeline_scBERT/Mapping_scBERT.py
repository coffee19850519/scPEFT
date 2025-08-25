import argparse
from colorama import init, Fore
import numpy
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from performer_pytorch.performer_pytorch_finetune import PerformerLM
import scanpy as sc
import pickle as pkl
import warnings
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.stats import mode
import sklearn
import torch
import numpy as np
import anndata as ad
from tqdm import tqdm
import faiss
from sklearn.metrics import f1_score, balanced_accuracy_score
import os
from Benckmark_utils import get_similar_vectors, SCDataset

faiss_imported = True
warnings.filterwarnings("ignore", category=ResourceWarning)

SEED = 2021
BATCH_SIZE = 50
SEQ_LEN = 16906 + 1
PATIENCE = 10
CLASS = 7
POS_EMBED_USING = True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='NSCLC', help='NSCLC/COVID-19/ms')
parser.add_argument("--data_path", type=str, default='../data/celltype_annoration/', help='Path of data for finetune.')

args = parser.parse_args()
data_path = args.data_path
dataset_name = args.dataset_name
device = torch.device("cuda:3")


class scBERT_preprocess():
    def __init__(self, panglao_adata, ref_adata, query_adata):
        super().__init__()
        self.panglao_adata = panglao_adata
        self.ref_adata = ref_adata
        self.query_adata = query_adata

    def align(self, adata):
        counts = sparse.lil_matrix((adata.X.shape[0], self.panglao_adata.X.shape[1]), dtype=np.float32)
        ref = self.panglao_adata.var_names.tolist()
        obj = adata.var_names.tolist()

        # Logging match gene nums
        match_gene = list(set(obj) & set(ref))
        print(f'scBERT match gene {len(match_gene)}/{len(obj)}, scBERT vocab size is {len(ref)}')

        for i in range(len(ref)):
            if ref[i] in obj:
                loc = obj.index(ref[i])
                counts[:, i] = adata.X[:, loc]

        counts = counts.tocsr()
        new = ad.AnnData(X=counts)
        new.var_names = ref
        new.obs_names = adata.obs_names
        new.obs = adata.obs
        new.uns = self.panglao_adata.uns

        return new

    def log1p_norm(self, adata):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata, base=2)

        return adata

    def norm_(self, adata):
        sc.pp.normalize_total(adata, target_sum=1e4)

        return adata

    def __call__(self, data_is_raw=False):
        # align
        ref_query = list(map(lambda dataset: self.align(dataset), [self.ref_adata, self.query_adata]))

        # norm
        norm_fn = self.log1p_norm if data_is_raw else self.norm_
        aligned_ref, aligned_query = map(lambda dataset: norm_fn(dataset), ref_query)

        # return
        return aligned_ref, aligned_query


# data path
for fold_idx in range(5):
    ref_path = f'{data_path}/{dataset_name}/{str(fold_idx)}/{dataset_name}_train{str(fold_idx)}.h5ad'
    query_path = f'{data_path}/{dataset_name}/{str(fold_idx)}/{dataset_name}_test{str(fold_idx)}.h5ad'
    model_path = './panglao_pretrain.pth'

    # read file as adata
    ref_adata = sc.read_h5ad(ref_path)
    query_adata = sc.read_h5ad(query_path)

    # panglao = sc.read_h5ad('./panglao_human.h5ad')
    panglao = sc.read_h5ad('/mnt/Data1/frx/scBERT_rollback/data10x/panglao_human.h5ad')

    if dataset_name == 'ms':
        data_is_raw = False
        cell_type_key = 'celltype'
        ref_adata.var_names = ref_adata.var["gene_name"]
        query_adata.var_names = query_adata.var["gene_name"]
    elif dataset_name == 'COVID-19':
        data_is_raw = True
        cell_type_key = 'cell_type'
    elif dataset_name == 'NSCLC':
        data_is_raw = True
        cell_type_key = 'cell_type'

    # read file as adata
    ref_adata = sc.read_h5ad(ref_path)
    query_adata = sc.read_h5ad(query_path)

    # process
    preprocess = scBERT_preprocess(panglao, ref_adata, query_adata)
    ref_adata, query_adata = preprocess(data_is_raw=data_is_raw)

    ref_data = ref_adata.X
    query_data = query_adata.X
    if isinstance(ref_data, numpy.ndarray):
        ref_data = sparse.csr_matrix(ref_data)

    if isinstance(query_data, numpy.ndarray):
        query_data = sparse.csr_matrix(query_data)

    ref_dataset = SCDataset(ref_data, CLASS)
    ref_loader = DataLoader(ref_dataset, batch_size=BATCH_SIZE)
    query_dataset = SCDataset(query_data, CLASS)
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE)

    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=True,
    )

    path = model_path
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], False)
    model = model.to(device)
    model.eval()

    # extract reference cell emb
    ref_embs = []
    with torch.no_grad():
        for (data_v) in tqdm(ref_loader, desc="Embedding Cell"):
            data_v = data_v.to(device)
            logits = model(data_v, return_encodings=True)

            for i in range(logits.shape[0]):
                # mean pooling
                pooling_emb = torch.mean(logits[i], dim=-1)[0:16906]
                ref_embs.append(pooling_emb.detach().cpu().numpy())
        # updata ref adata.X with cell embedding
        ref_adata.X = np.array(ref_embs)

    # extract query cell emb
    query_embs = []
    with torch.no_grad():
        for (data_v) in tqdm(query_loader, desc="Embedding query cell"):
            data_v = data_v.to(device)
            logits = model(data_v, return_encodings=True)

            for i in range(logits.shape[0]):
                # mean pooling
                pooling_emb = torch.mean(logits[i], dim=-1)[0:16906]
                query_embs.append(pooling_emb.detach().cpu().numpy())

        # updata query adata.X with cell embedding
        query_adata.X = np.array(query_embs)

    # %%
    ref_cell_embeddings = ref_adata.X
    test_emebd = query_adata.X

    k = 10  # number of neighbors

    if faiss_imported:
        # Declaring index, using most of the default parameters from
        index = faiss.IndexFlatL2(ref_cell_embeddings.shape[1])
        index.add(ref_cell_embeddings)

        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        distances, labels = index.search(test_emebd, k)

    idx_list = [i for i in range(test_emebd.shape[0])]
    preds = []
    for k in idx_list:
        if faiss_imported:
            idx = labels[k]
        else:
            idx, sim = get_similar_vectors(test_emebd[k][np.newaxis, ...], ref_cell_embeddings, k)
        pred = mode(ref_adata.obs[cell_type_key][idx], axis=0)
        preds.append(pred[0][0])

    gt = query_adata.obs[cell_type_key].to_numpy()
    train_label_dict, train_label = np.unique(np.array(ref_adata.obs[cell_type_key]), return_inverse=True)
    truths = ref_adata.obs[cell_type_key].tolist()

    f1 = f1_score(gt, preds, average='macro')
    accuracy = balanced_accuracy_score(gt, preds)

    print(f'    ==  F1 Score: {f1:.6f} | Acc: {accuracy * 100:.4f}%  ==')
