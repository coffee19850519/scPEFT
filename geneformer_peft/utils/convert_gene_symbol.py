import scanpy as sc
import pickle
import os
from pathlib import Path

with open('../geneformer/gene_name_id_dict.pkl', 'rb') as f:
    name_id_dict = pickle.load(f)


def convert_gene_symbol(adata, dataset):
    if dataset in ['MergedMonkey', "mouse_115746"]:
        genes_key = "gene_name"
    if dataset == 'elegans':
        genes_key = "orthogene"

    gene_names = adata.var[genes_key].tolist()

    gene_ids = []
    name_in_ids = []

    for index, gene_name in enumerate(gene_names):
        if gene_name in list(name_id_dict.keys()):
            name_in_ids.append(1)
            gene_ids.append(name_id_dict[gene_name])
        else:
            name_in_ids.append(0)
            gene_ids.append(None)

    adata.var["ensembl_id"] = gene_ids
    adata.var["name_in_ids"] = name_in_ids

    adata = adata[:, adata.var["name_in_ids"] > 0]
    del adata.var["name_in_ids"]

    return adata


Repositioning_Path = "/mnt/Data3/22frx/CrossS_Cache"

data_path = "/mnt/Data5/23zxy/scPEFT-main - A6000/newdata/"
# datasets = ["MergedMonkey", "elegans", "mouse_115746"]
datasets = ["mouse_115746"]
for dataset in datasets:
    current_dataset_path = os.path.join(data_path, dataset)
    list_dirs = sorted(os.listdir(current_dataset_path))
    for list_dir in list_dirs:
        current_fold_path = os.path.join(current_dataset_path, list_dir)
        train_val_test = sorted(os.listdir(current_fold_path))
        for item in train_val_test:
            adata = sc.read(os.path.join(current_fold_path, item))
            adata = convert_gene_symbol(adata, dataset)

            restorage_path = Path(os.path.join(Repositioning_Path, dataset, list_dir))
            restorage_path.mkdir(parents=True, exist_ok=True)

            adata.write_h5ad(os.path.join(restorage_path, item))

print("")
