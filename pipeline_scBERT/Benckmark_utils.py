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
from performer_pytorch import PerformerLM
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
# import faiss
from sklearn.metrics import f1_score, balanced_accuracy_score

faiss_imported = True
warnings.filterwarnings("ignore", category=ResourceWarning)


class SCDataset(Dataset):
    def __init__(self, data, CLASS):
        super().__init__()
        self.data = data
        self.CLASS = CLASS

    def __getitem__(self, index):
        rand_start = index
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (self.CLASS - 2)] = self.CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        return full_seq

    def __len__(self):
        return self.data.shape[0]


class scBERT_preprocess():
    def __init__(self, panglao_adata, need_process_adata):
        super().__init__()
        self.panglao_adata = panglao_adata
        self.need_process_adata = need_process_adata

    def align(self, adata):
        counts = sparse.lil_matrix((adata.X.shape[0], self.panglao_adata.X.shape[1]), dtype=np.float32)
        ref = self.panglao_adata.var_names.tolist()
        obj = adata.var_names.tolist()

        # Logging match gene nums
        mask = [False] * len(ref)
        match_gene = list(set(obj) & set(ref))
        # print(match_gene)
        print(f'scBERT match gene {len(match_gene)}/{len(obj)}, scBERT vocab size is {len(ref)}')

        for i in range(len(ref)):
            if ref[i] in obj:
                loc = obj.index(ref[i])
                counts[:, i] = adata.X[:, loc]
                mask[i] = True

        counts = counts.tocsr()
        new = ad.AnnData(X=counts)
        new.var_names = ref
        new.obs_names = adata.obs_names
        new.obs = adata.obs
        new.uns = self.panglao_adata.uns
        new.uns["gene_mask"] = mask
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
        ref_query = list(map(lambda dataset: self.align(dataset), [self.need_process_adata]))

        # norm
        # norm_fn = self.log1p_norm if data_is_raw else self.norm_
        # aligned_adata = list(map(lambda dataset: norm_fn(dataset), ref_query))[0]

        # return
        # return aligned_adata[0]
        return ref_query[0]


# Those functions are only used when faiss is not installed
def l2_sim(a, b):
    sims = -np.linalg.norm(a - b, axis=1)
    return sims


def get_similar_vectors(vector, ref, top_k=10):
    # sims = cos_sim(vector, ref)
    sims = l2_sim(vector, ref)

    top_k_idx = np.argsort(sims)[::-1][:top_k]
    return top_k_idx, sims[top_k_idx]
