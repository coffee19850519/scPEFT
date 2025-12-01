# scPEFT: Harnessing the Power of Single-Cell Large Language Models with Parameter-Efficient Fine-Tuning 

This is the official repository for **Harnessing the Power of Single Cell Large Language Models with Parameter Efficient
Fine-Tuning using scPEFT**. To reproduce the results from the paper, please visit [scPEFT_reproduction](https://github.com/coffee19850519/scPEFT_reproduction).

**:exclamation: 9/25/2025 UPDATE** We have released the reproduction for benchmarking the linear model on perturbation prediction. Please see [here](https://github.com/coffee19850519/scPEFT_reproduction/blob/main/README.md#exclamation-9252025-update-perturbation-prediction-with-the-linear-model) for more details.

**:fire:UPDATE**: We have released several new reproductions for the revised manuscript. Please see [scPEFT_reproduction](https://github.com/coffee19850519/scPEFT_reproduction) for more details.


[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.01.27.577455v1)
&nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

[![DOI](https://zenodo.org/badge/868709362.svg)](https://doi.org/10.5281/zenodo.17781911)

## Overview
we propose scPEFT, a framework that integrates Parameter-Efficient Fine-Tuning (PEFT) techniques into scLLMs to calibrate them for specialized use cases. Unlike traditional finetuning, which modifies the entire model, scPEFT employs low-dimensional, learnable, and pluggable adapters to customize scLLMs in a separate, reduced-dimensional subspace. The critical role of these proxy adapters is to estimate a ‘model delta’ (standing for changes in some model parameters) for context alignment under the guidance of task-specific objective functions and limited custom data. During the adaptation process, the original scLLM parameters are frozen to preserve pre-learned biological knowledge, while only the smaller adapter parameters are updated. This design reduces the complexity of domain adaptation, enabling higher performance with fewer resources than traditional finetuning strategies of scLLMs in out-of-context scenarios.

![overview](https://github.com/coffee19850519/scPEFT/blob/main/img/overview.jpg)

## Installation

scPEFT works with Python >= 3.7.13. scPEFT is available on PyPI. To install scPEFT, run the following command:

```bash
pip install scpeft
```

For developing, run the following command:

```
git clone https://github.com/coffee19850519/scPEFT
cd scPEFT
```

**Note**: scPEFT is currently built on top of [scGPT](https://github.com/bowang-lab/scGPT), [scBERT](https://github.com/TencentAILabHealthcare/scBERT), [scFoundation](https://github.com/biomap-research/scFoundation/) and [Geneformer](https://huggingface.co/ctheodoris/Geneformer).
Please follow their installation instructions to ensure all necessary versioned dependencies are installed. We provide a [requirements. ymal](https://github.com/SELECT-FROM/scPEFT/blob/main/requirements.yaml) file for the environment in which scPEFT was developed.

## Get Started

1. Download the backbone
   model, e.g., [scGPT model checkpoint](https://github.com/bowang-lab/scGPT/blob/main/README.md#pretrained-scgpt-model-zoo)
   and place it at e.g., `work_dir/scPEFT/save`. We recommend using
   the [whole-human](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) model for
   most applications by default, which pretrained on 33 million normal human cells.

2. The tutorials of scPEFT for downstream tasks
   in  [tutorial_peft](https://github.com/coffee19850519/scPEFT/tree/main/tutorial_peft). Here are the links to the
   downstream tasks and tutorials mentioned in our article

  | Backbone| Downstream task           | Link                                                                                                                                           |
  |:-------------------------- |:--------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
  |scGPT | cell type identification  | [Tutorial_Identification.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_Identification.ipynb)                |
  |scGPT | batch correction          | [Tutorial_BatchCorrection.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_BatchCorrection.ipynb)                 |
  |scGPT | perturbation              | [Tutorial_Perturbation.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_Perturbation.ipynb)                       |
  |scGPT | cell population discovery | [Tutorial_CellPopulationDiscovery.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_CellPopulationDiscovery.ipynb) |
  |scGPT | marker gene detection     | [Tutorial_MarkerGeneDetection.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_MarkerGeneDetection.ipynb)         |
  |scFoundation | cell type identification  | [Tutorial_identification_scFoundation.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_identification_scFoundation.ipynb)               |
  |scFoundation | perturbation              | [Tutorial_Perturbation_scFoundation.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_perturbation_scFoundation.ipynb)                       |
  |Geneformer | cell type identification  | [Tutorial_identification_Geneformer.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_identification_Geneformer.ipynb)              |
  |scBERT | cell type identification  | [Tutorial_identification_scBERT.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_identification_scBERT.ipynb)               |

## To-do-list

- [x] Publish to pypi
- [x] Adapting scPEFT for native-attention
- [ ] Release scripts for more downstream tasks
- [x] Release scripts for more scLLM backbones
- [ ] Adapting scPEFT for flash-attention
- [ ] Only retain PEFT-related parameters when saving peft-model weights.

## Contributing

We greatly welcome contributions to scPEFT. Please submit a pull request if you have any ideas or bug fixes. We also
welcome any issues you encounter while using scPEFT.

## Built With

We sincerely thank the authors of following open-source projects:

- [scGPT](https://github.com/bowang-lab/scGPT)
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- [scBERT](https://github.com/TencentAILabHealthcare/scBERT)
- [scFoundation](https://github.com/biomap-research/scFoundation/)
- [scanpy](https://github.com/scverse/scanpy)
- [scib](https://github.com/theislab/scib)
- [pytorch](https://github.com/pytorch/pytorch)

## Citing scPEFT

```bibtex
@article {He2025.04.21.649754,
	author = {He, Fei and Fei, Ruixin and Krull, Jordan E. and Zhang, Xinyu and Gao, Mingyue and Su, Li and Chen, Yibo and Yu, Yang and Li, Jinpu and Jin, Baichuan and Chang, Yuzhou and Ma, Anjun and Ma, Qin and Xu, Dong},
	title = {Harnessing the Power of Single-Cell Large Language Models with Parameter Efficient Fine-Tuning using scPEFT},
	year = {2025},
	doi = {10.1101/2025.04.21.649754},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/04/23/2025.04.21.649754},
	journal = {bioRxiv}
}

```
