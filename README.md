# Towards Principled Graph Transformers

[![arXiv](https://img.shields.io/badge/arXiv-2401.10119-b31b1b.svg)](https://arxiv.org/abs/2401.10119)
[![pytorch](https://img.shields.io/badge/PyTorch_2.1.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pyg](https://img.shields.io/badge/PyG_2.4+-3C2179?logo=pyg&logoColor=#3C2179)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3.2-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

Code for our paper [Towards Principled Graph Transformers](https://arxiv.org/abs/2401.10119). 

> Our implementation of the `EdgeAttention` is built on top of the code provided in Bergen et al. 2021, Systematic Generalization with Edge Transformers, available at https://github.com/bergen/EdgeTransformer.

## Install
We recommend to use the package manager [`conda`](https://docs.conda.io/en/latest/). Once installed run
```bash
conda create -n towards-principled-gts python=3.10
conda activate towards-principled-gts
```
Install all dependencies via
```bash
pip install -e .
```

## Configuration
We use [`hydra`](https://hydra.cc) for configuring experiments. See [here](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) for a tutorial on the `hydra` override syntax.

> NOTE: By default, logging with `wandb` is disabled. To enable it set `wandb_project` in the command line. Optionally, set `wandb_entity` and `wandb_name` to configure your entity and run name, respectively.

## Expressivity
For the BREC benchmark, run
```bash
python expressivity/main.py root=/path/to/data/root
```
respectively, where `/path/to/data/root` specifies the path to your data folder. This folder will be created if it does not exist.

## Molecular regression
To run the ZINC, Alchemy or QM9 dataset, run
```bash
python molecular-regression/[zinc|alchemy|qm9].py root=/path/to/data/root
```
respectively, where `/path/to/data/root` specifies the path to your data folder. This folder will be created if it does not exist.

## CLRS-30
For the CLRS experiments see our dedicated fork at [https://github.com/ksmdnl/clrs](https://github.com/ksmdnl/clrs).
