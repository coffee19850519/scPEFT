"""
Geneformer tokenizer.

Input data:
Required format: raw counts scRNAseq data without feature selection as .loom file
Required row (gene) attribute: "ensembl_id"; Ensembl ID for each gene
Required col (cell) attribute: "n_counts"; total read counts in that cell
Optional col (cell) attribute: "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria
Optional col (cell) attributes: any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below

Usage:
  from geneformer import TranscriptomeTokenizer
  tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("loom_data_directory", "output_directory", "output_prefix")
"""
from __future__ import annotations
from typing import Literal
import pickle
from pathlib import Path

import logging

import warnings

import requests


warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import os
import anndata as ad
import loompy as lp
import numpy as np
from datasets import Dataset
from scipy.sparse import csr_matrix
logger = logging.getLogger(__name__)

GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary.pkl"


# gene vector is normed gene, which shape is (1,gene_num)
# gene tokens is the token id of every gene
def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]  # all nonzero exp index
    # sort by median-scaled gene values 从大到小的把这些值的index找到
    sorted_indices = np.argsort(-gene_vector[nonzero_mask])  # the index of value from large to small
    # tokenize 从token id里面先把0的位置去掉，然后对应着从大到小的去取对应的token id 去表示对应的gene
    sentence_tokens = gene_tokens[nonzero_mask][sorted_indices]  # the token id of value from large to small
    return sentence_tokens


class TranscriptomeTokenizer:
    def __init__(
            self,
            custom_attr_name_dict=None,
            nproc=1,
            gene_median_file=GENE_MEDIAN_FILE,
            token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset.
            Keys are the names of the attributes in the loom file.
            Values are the names of the attributes in the dataset.
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    # def tokenize_data(self, loom_data_directory, output_directory, output_prefix):
    #     """
    #     Tokenize .loom files in loom_data_directory and save as tokenized .dataset in output_directory.
    #
    #     Parameters
    #     ----------
    #     loom_data_directory : Path
    #         Path to directory containing loom files
    #     output_directory : Path
    #         Path to directory where tokenized data will be saved as .dataset
    #     output_prefix : str
    #         Prefix for output .dataset
    #     """
    #     tokenized_cells, cell_metadata = self.tokenize_files(Path(loom_data_directory))
    #     tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata)
    #
    #     output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
    #     tokenized_dataset.save_to_disk(output_path)

    def tokenize_data(
            self,
            adata_scale: None,
            dataset:str,
            data_directory: Path | str,
            output_directory: Path | str,
            output_prefix: str,
            file_format: Literal["loom", "h5ad"] = "loom",
    ):
        """
        Tokenize .loom files in loom_data_directory and save as tokenized .dataset in output_directory.

        Parameters
        ----------
        loom_data_directory : Path
            Path to directory containing loom files or anndata files
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        file_format : str
            Format of input files. Can be "loom" or "h5ad".
        """
        tokenized_cells, cell_metadata = self.tokenize_files(adata_scale,dataset,
                                                             Path(data_directory), file_format
                                                             )
        tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata)

        if adata_scale is not None:
            output_path = (Path(output_directory) / (output_prefix + '_scale{:.2f}'.format(adata_scale))).with_suffix(
                ".dataset")
        else:
            output_path = (Path(output_directory) / output_prefix/'_all').with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path)

    # def tokenize_files(self, loom_data_directory):
    #     tokenized_cells = []
    #     if self.custom_attr_name_dict is not None:
    #         loom_cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
    #         cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}
    #
    #     # loops through directories to tokenize .loom files
    #     file_found = 0
    #     for loom_file_path in loom_data_directory.glob("*.loom"):
    #         file_found = 1
    #         print(f"Tokenizing {loom_file_path}")
    #         file_tokenized_cells, file_cell_metadata = self.tokenize_file(
    #             loom_file_path
    #         )
    #         tokenized_cells += file_tokenized_cells
    #         if self.custom_attr_name_dict is not None:
    #             for k in loom_cell_attr:
    #                 cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
    #         else:
    #             cell_metadata = None
    #
    #     if file_found == 0:
    #         logger.error(
    #             f"No .loom files found in directory {loom_data_directory}.")
    #         raise
    #     return tokenized_cells, cell_metadata

    def tokenize_file(self, loom_file_path):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )[0]
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["ensembl_id"][coding_miRNA_loc]
                ]
            )
            coding_miRNA_ids = data.ra["ensembl_id"][coding_miRNA_loc]
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except AttributeError:
                var_exists = False
            else:
                var_exists = True

            if var_exists is True:
                filter_pass_loc = np.where(
                    [True if i == 1 else False for i in data.ca["filter_pass"]]
                )[0]
            elif var_exists is False:
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                filter_pass_loc = np.array([i for i in range(data.shape[1])])

            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                        subview[:, :]
                        / subview.ca.n_counts
                        * 10_000
                        / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None

        return tokenized_cells, file_cell_metadata

    def tokenize_files(
            self, adata_scale,dataset, data_directory, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            loom_cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        # loops through directories to tokenize .loom files
        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        tokenize_file_fn = (
            self.tokenize_file if file_format == "loom" else self.tokenize_anndata
        )
        for file_path in data_directory.glob("*.{}".format(file_format)):
            # tokenized_cells=[]
            # cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}
            file_found = 1
            print(f"Tokenizing {file_path}")
            file_tokenized_cells, file_cell_metadata = tokenize_file_fn(dataset=dataset,adata_scale=adata_scale, adata_file_path=file_path)

            tokenized_cells =file_tokenized_cells
            if self.custom_attr_name_dict is not None:
                for k in loom_cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] = file_cell_metadata[k]
            else:
                cell_metadata = None

            tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata)
            output_path = (
                        Path('/media/fei/Data/zxy/Geneformer 3090/data_geneformer/elegans') / os.path.basename(file_path).split('.')[0].format(adata_scale)).with_suffix(".dataset")
            tokenized_dataset.save_to_disk(output_path)

        if file_found == 0:
            logger.error(
                f"No .{file_format} files found in directory {data_directory}.")
            raise

        return tokenized_cells, cell_metadata

    def gene_name_to_ensembl_id(gene_name):
        base_url = "https://rest.ensembl.org"
        endpoint = "/lookup/symbol/homo_sapiens/{}?content-type=application/json".format(gene_name)
        url = base_url + endpoint

        response = requests.get(url)
        if response.ok:
            data = response.json()
            if 'id' in data:
                ensembl_id = data['id']
                return ensembl_id
            else:
                return "Gene not found"
        else:
            return "Request failed"

    def tokenize_anndata(self, dataset,adata_scale, adata_file_path):

        adata = ad.read(adata_file_path)

        if "ms_" in str(adata_file_path):
            adata.var["ensembl_id"] = adata.var["index_column"]
            adata.obs["cell_type"] = adata.obs["celltype"]
        if "mouse_115746" in str(adata_file_path):
            adata.obs["cell_type"] = adata.obs["cell_class"]
        if "MergedMonkey" in str(adata_file_path):
            adata.obs["cell_type"] = adata.obs["CellType"]
        if "zheng68k" in str(adata_file_path):
            adata.obs["cell_type"] = adata.obs["celltype"]
        if "elegans" in str(adata_file_path):
            adata.obs["cell_type"] = adata.obs["tissue_name"]
        # if "norman" or "adamson" in str(adata_file_path):
        #     adata.var["ensembl_id"] =adata.var_names
        file_cell_metadata = {
            attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
        }

        # if dataset in ['COVID-19','mouse_115746' , 'elegans']:
        #     gene_names = adata.var_names.tolist()
        #     gene_names_list = []
        #     print(f'all gene num is {adata.shape[-1]}')
        #
        #     for index, gene_id in enumerate(gene_names):
        #         ensembl_id = gene_name_to_ensembl_id(gene_id)
        #         if ensembl_id:
        #             print(f"The Ensembl ID for gene {gene_id} is {ensembl_id},this is No.{index}")
        #             gene_names_list.append(ensembl_id)
        #         else:
        #             print(f"No Ensembl ID found for gene {gene_id},this is No.{index}")
        #             gene_names_list.append(None)
        #     adata.var["ensembl_id"]=gene_names_list
        # if dataset in ['COVID-19','mouse_115746' , 'elegans']:
        #     mg = MyGeneInfo()
        #     gene_names = adata.var['gene_name'].values
        #     gene_info = mg.querymany(gene_names, scopes='symbol', fields='ensembl.gene', species='human')
        #
        #     ensembl_id_dict = {}
        #     for gene in gene_info:
        #         if 'ensembl' in gene and len(gene['ensembl'])==1:
        #             ensembl_id_dict[gene['query']] = gene['ensembl']['gene']
        #         else:
        #             ensembl_id_dict[gene['query']] = None
        #
        #     ensembl_ids = [ensembl_id_dict[gene_name] for gene_name in gene_names]
        #     adata.var['ensembl_id'] = ensembl_ids

        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in adata.var["ensembl_id"]]
        )[0]
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in adata.var["ensembl_id"][coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = adata.var["ensembl_id"][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        try:
            adata.obs["filter_pass"]
        except KeyError:
            var_exists = False
        else:
            var_exists = True

        if var_exists is True:
            filter_pass_loc = np.where(
                [True if i == 1 else False for i in adata.obs["filter_pass"]]
            )[0]
        elif var_exists is False:
            print(
                f"{adata_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(adata.shape[0])])

        tokenized_cells = []
        adata_filter = adata[
            filter_pass_loc, coding_miRNA_loc  # filter cells and genes
        ]

        X_norm = (adata_filter.X / adata.X.sum(1).reshape(-1,1) * 10_000 / norm_factor_vector)
        X_norm = csr_matrix(X_norm)

        tokenized_cells += [
            tokenize_cell(X_norm[i, ...].A.flatten(), coding_miRNA_tokens)
            for i in range(X_norm.shape[0])
        ]

        # add custom attributes for subview to dict
        for k in file_cell_metadata.keys():
            file_cell_metadata[k] += adata_filter.obs[k].tolist()

        return tokenized_cells, file_cell_metadata

    def create_dataset(self, tokenized_cells, cell_metadata):
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # create dataset
        output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        def truncate(example):
            example["input_ids"] = example["input_ids"][0:2048]
            return example

        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)

        # measure lengths of dataset
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example

        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )

        return output_dataset_truncated_w_length
