# CL-GraphTrans-DTA

This repository contains the source code for the paper: **"A Contrastive Learning-enhanced Graph Transformer Framework for Drug-Target Affinity Prediction"**.

## 🚀 Introduction

CL-GraphTrans-DTA is a deep learning framework that combines **Graph Attention Networks (GATv2)** for molecular representation and **Transformers** for protein sequence encoding. It introduces a **Contrastive Learning** module to align the latent spaces of drugs and targets, significantly improving generalization on unseen data.

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric (PyG)
- RDKit
- Lifelines

## 📥 Installation

```bash
# 1. Clone the repository
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyG dependencies (Adjust based on your CUDA version)
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
