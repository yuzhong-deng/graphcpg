import multiprocessing as mp
import numpy as np
import networkx as nx
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as ssp
import bisect
import seaborn as sns
from datamodules import CpGGraphImputationDataModule
from graphcpg import CpGGraph
# setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
segment_size = 21
batch_size = 10240
n_workers = 12
res_dir = "/home/bruce/Mylab/Bio/graphcpg/root/ckpt"
data_path = "/home/bruce/Mylab/Bio/data/"
dataset_dict = {"Hemato":"y_Hemato.npz"}
ckpt_dict =    {"Hemato":"/Hemato_epoch=7-step=23536.ckpt"}

y = np.load(data_path + dataset_dict["Hemato"])
cell_num = y["chr1"].shape[1]

model_ckpt = res_dir + ckpt_dict["Hemato"]
model = CpGGraph.load_from_checkpoint(model_ckpt)
datamodule = CpGGraphImputationDataModule(y, segment_size=segment_size,
                                batch_size=batch_size, n_workers=n_workers,
                                cell_nums=False)
datamodule.setup()
all_graphs = datamodule.all
model.eval()
model.to(device)
imputation_dataloader = DataLoader(all_graphs, 
                            batch_size=batch_size, shuffle=False,
                            pin_memory=True)
ys_imputed_combined = {}
used_chrs = y.files
for chrom in used_chrs:
    ys_imputed_combined[chrom] = y[chrom].astype(np.half)
    ys_imputed_combined[chrom] = y[chrom][:, :cell_num].astype(np.half)
for data in tqdm(imputation_dataloader):
    data = data.to(device)
    r_hat = torch.sigmoid(model(data)).detach().cpu().numpy().astype(np.half)
    chr_ind = data.chr
    row_locus = data.m
    col_cell = data.n
    for ind in range(len(r_hat)):
        ys_imputed_combined[used_chrs[chr_ind[ind]]][row_locus[ind], col_cell[ind]] = r_hat[ind]
print('Writing combined files ...')
np.savez_compressed("y_imputed.npz", **ys_imputed_combined)
print('end')