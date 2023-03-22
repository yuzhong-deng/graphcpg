# GraphCpG

| Dataset      | Cell Num | (AUROC) CaMelia | (AUROC) DeepCpG | (AUROC) CpG Transformer | (AUROC) GraphCpG |
|--------------|----------|:---------------:|:---------------:|:-----------------------:|:----------------:|
| HCC          |    25    |      97.11      |      96.01      |        **97.56**        |       96.97      |
| MBL          |    30    |      89.36      |      87.12      |        **92.05**        |       89.73      |
| Hemato       |   122    |      87.68      |      88.26      |          89.56          |     **89.77**    |
| Neuron-Mouse |   690    |      91.13      |      88.59      |          90.87          |     **91.75**    |
| Neuron-Homo  |   780    |      92.98      |      90.06      |          92.31          |     **93.2**     |

# Visualization of locus-aware neighboring subgraphs from HCC
![Image text](https://github.com/yuzhong-deng/graphcpg/blob/9353ba350eaac88b10bc77c7a3c031f475456c27/visualization_HCC_visual_prediction.png)


# Folders
- Data
  - Neuron-Mouse
  - Neuron-Homo
- Model
  - graphcpg
    - training
    - imputation
    - visualization
 
# Environment and Packages

- python 3.9
- cuda 11.1
- pytorch 1.9.1+cu111
- torhcvision 0.10.1+cu111
- torchaudio 0.9.1
- pytorch_geometric
- torch-sparse
- torch-geometric
- if error (module 'distutils'), please install setuptools 59.5.0


