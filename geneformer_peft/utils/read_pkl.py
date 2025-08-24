import scanpy as sc
import pickle
from scipy.sparse import issparse

with open('../geneformer/gene_name_id_dict.pkl', 'rb') as f:
    name_id_dict = pickle.load(f)

adata = sc.read("/media/fei/Data/frx/Geneformer_unsupervise/example_py/data/tonsil/GCBC.h5ad")
gene_names = adata.var["name"].tolist()

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

adata.obs["cell_type"] = adata.obs["annotation_20230508"]

# train_data = adata.X.A if issparse(adata.X) else adata.X
# adata.X = train_data

print("data")
