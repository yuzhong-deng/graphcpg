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


# Files
- Model
  - graphcpg
    - training
    - imputation
    - visualization
- Data
  - Neuron-Mouse
  - Neuron-Homo
- Analysis
    - hierarchical analysis
    - differential methylation analysis
 
# Environment and Packages

- python 3.9
- cuda 11.1
- pytorch 1.9.1+cu111
- torhcvision 0.10.1+cu111
- torchaudio 0.9.1
- pytorch_geometric
- torch-sparse
- torch-geometric
- pytorch-lightning 1.7.7

if error (module 'distutils'), please install setuptools 59.5.0

requirement.txt is also provided for reference, but compatibility still depends on the devices.

pytorch-lightning version incompatibility may cause extremely long training time and degraded result. (millions of thanks for colleagues asking questions related to this problem by emailing and in issues)

# Usage
Enter the model folder

`python train_graph_cpg.py`

# Citation
[GraphCpG: Imputation of Single-cell Methylomes Based on Locus-aware Neighboring Subgraphs](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btad533/7255916?login=true)
```bibtex
@article{Deng2023GraphCpGIO,
  title={GraphCpG: imputation of single-cell methylomes based on locus-aware neighboring subgraphs},
  author={Yuzhong Deng and Jianxiong Tang and Jiyang Zhang and Jianxiao Zou and Que Zhu and Shicai Fan},
  journal={Bioinformatics},
  year={2023},
  volume={39},
  url={https://api.semanticscholar.org/CorpusID:261381228}
}
```
# License

- MIT license
