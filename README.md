# scPEFT

This is the official repository for **Harnessing the Power of Single Cell Large Language Models with Parameter Efficient
Fine-Tuning using scPEFT**. To reproduce the results from the paper, please visit [scPEFT_reproduction](https://github.com/coffee19850519/scPEFT_reproduction).

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.01.27.577455v1)
&nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

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

**Note**: scPEFT is currently built on top of [scGPT](https://github.com/bowang-lab/scGPT), [scBERT](https://github.com/TencentAILabHealthcare/scBERT), and [Geneformer](https://huggingface.co/ctheodoris/Geneformer).
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

   | Downstream task           | Link                                                                                                                                           |
   |:--------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
   | cell type identification  | [Tutorial_Identification.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_Identification.ipynb)                |
   | batch correction          | [Tutorial_BatchCorrection.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_BatchCorrection.ipynb)                 |
   | perturbation              | [Tutorial_Perturbation.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_Perturbation.ipynb)                       |
   | cell population discovery | [Tutorial_CellPopulationDiscovery.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_CellPopulationDiscovery.ipynb) |
   | marker gene detection     | [Tutorial_MarkerGeneDetection.ipynb](https://github.com/coffee19850519/scPEFT/blob/main/tutorial_peft/Tutorial_MarkerGeneDetection.ipynb)         |

## To-do-list

- [x] Publish to pypi
- [x] Adapting scPEFT for native-attention
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
- [scanpy](https://github.com/scverse/scanpy)
- [scib](https://github.com/theislab/scib)
- [pytorch](https://github.com/pytorch/pytorch)

## Citing scPEFT

```bibtex
@article {He2024.01.27.577455,
	author = {Fei He and Ruixin Fei and Mingyue Gao and Li Su and Xinyu Zhang and Dong Xu},
	title = {Parameter-Efficient Fine-Tuning Enhances Adaptation of Single Cell Large Language Model for Cell Type Identification},
	year = {2024},
	doi = {10.1101/2024.01.27.577455},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/01/30/2024.01.27.577455},
	journal = {bioRxiv}
}
```
