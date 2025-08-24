from geneformer import InSilicoPerturber
from geneformer import InSilicoPerturberStats
from torch_geometric.loader import DataLoader
from gears import PertData
from gears.gears import GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
# in silico perturbation in deletion mode to determine genes whose
# deletion in the dilated cardiomyopathy (dcm) state significantly shifts
# the embedding towards non-failing (nf) state
# data_name = "norman"
# split = "simulation"
# lr = 1e-4  # or 1e-4
# batch_size = 30
# eval_batch_size = 12
# epochs = 40
# if data_name == "norman":
#     perts_to_plot = ["SAMD1+ZBTB1"]
# elif data_name == "adamson":
#     perts_to_plot = ["KCTD16+ctrl"]
# pert_data = PertData("/mnt/Data1/zmr/Geneformer/example_py/data/")
# pert_data.load(data_name=data_name)
# pert_data.prepare_split(split=split, seed=1)
# pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
#

# rank(pert_data.dataloader['train_loader'])
isp = InSilicoPerturber(perturb_type="delete",
                        perturb_rank_shift=None,
                        genes_to_perturb="all",
                        combos=0,
                        #perturb genes individually (0) or in pairs (1)
                        anchor_gene=None,
                        #combos=1，as a anchor
                        model_type="Pretrained",
                        num_classes=0,
                        emb_mode="cell_and_gene",
                        cell_emb_style="mean_pool",
                        filter_data={"cell_type":["astrocyte","cortical layer 2-3 excitatory neuron A","cortical layer 5-6 excitatory neuron"]},
                        # cell_states_to_model={"disease":(["dcm"],["nf"],["hcm"])},
                        # cell_states_to_model={"disease":None},
                        max_ncells=500,
                        emb_layer=-1,
                        forward_batch_size=100,
                        nproc=16)
# outputs intermediate files from in silico perturbation
isp.perturb_data("/mnt/Data1/23zmr/Geneformer/geneformer-12L-30M/",
                 "/mnt/Data1/frx/Geneformer/example_py/data/ms_train.dataset",
                 "/mnt/Data1/23zmr/data_perturb/output",
                 "delete_")
ispstats = InSilicoPerturberStats(mode="mixture_model",
                                  genes_perturbed="all",
                                  combos=0,
                                  anchor_gene=None)
                                  # cell_states_to_model={"disease":(["dcm"],["nf"],["hcm"])})

# extracts data from intermediate files and processes stats to output in final .csv
ispstats.get_stats("/mnt/Data1/23zmr/data_perturb",
                   None,
                   "/mnt/Data1/23zmr/data_perturb/output",
                   "stats_")