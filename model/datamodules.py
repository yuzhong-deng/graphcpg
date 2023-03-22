from time import time
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import scipy.sparse as ssp
import bisect
from tqdm import tqdm

class CpGGraphDataModule(pl.LightningDataModule):#*Only IGMC
    def __init__(self, y, segment_size=21, 
                 val_keys=None, test_keys=None,
                 batch_size=2, n_workers=4, cell_nums=False, save_cell_id=False):
        super().__init__()
        self.save_hyperparameters(ignore=['y'])
        self.y = y
        self.segment_size = segment_size
        self.half_seg = int((segment_size - 1)/2)
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.batch_size = batch_size
        self.nw = n_workers
        if cell_nums:
            self.cell_num =cell_nums
        else:
            self.cell_num = y["chr1"].shape[1]
        self.save_cell_id = save_cell_id
    
    def setup(self):#, stage):
        time_start = time()
        val_chrs = self.val_keys
        test_chrs = self.test_keys
        train_chrs = [i for i in self.y.keys()]
        train_chrs = [i for i in train_chrs if i not in val_chrs]
        train_chrs = [i for i in train_chrs if i not in test_chrs]
        val_graphs_temp, val_chrs_length_dict = self.extract_subgraphs(val_chrs, "val", self.cell_num)
        train_graphs_temp, train_chrs_length_dict = self.extract_subgraphs(train_chrs, "train", self.cell_num)
        test_graphs_temp, test_chrs_length_dict = self.extract_subgraphs(test_chrs, "test", self.cell_num)
        time_end = time()
        print("Running Time："+str(time_end - time_start)+"s")
        self.val = CpGGraphDataset(val_graphs_temp, val_chrs_length_dict, self.segment_size, self.save_cell_id)
        self.train = CpGGraphDataset(train_graphs_temp, train_chrs_length_dict, self.segment_size, self.save_cell_id)
        self.test = CpGGraphDataset(test_graphs_temp, test_chrs_length_dict, self.segment_size, self.save_cell_id)
        

    def extract_subgraphs(self, chr_names, dataset_type, cell_nums):

        chrs_graphs_temp = []
        chr_id = 0
        chrs_length_dict = [0]
        chr_length = 0
        for chr_name in tqdm(chr_names):
            y_temp = self.y[chr_name][:, :cell_nums]
            #padding methylation matrix[y_temp]
            y_temp_add = np.full((self.half_seg, y_temp.shape[1]), -1)
            y_temp = np.concatenate((y_temp_add, y_temp, y_temp_add))
            y_temp_csr = ssp.csr_matrix(y_temp + 1)
            chr_length += len(y_temp_csr.data)
            chrs_length_dict += [chr_length]
            chrs_graphs_temp += [y_temp_csr]
            chr_id + 1
        return chrs_graphs_temp, chrs_length_dict


    def train_dataloader(self):
        return DataLoader(self.train, num_workers=self.nw,
                            batch_size=self.batch_size, shuffle=True,
                            pin_memory=True, persistent_workers=True, prefetch_factor=10)

    def val_dataloader(self):
        return DataLoader(self.val, num_workers=self.nw,
                            batch_size=self.batch_size, shuffle=False,
                            pin_memory=True, persistent_workers=True, prefetch_factor=10)

    def test_dataloader(self):
        return DataLoader(self.test, num_workers=self.nw,
                            batch_size=self.batch_size, shuffle=False,
                            pin_memory=True, persistent_workers=True, prefetch_factor=10)

class CpGGraphDataset(Dataset):#*Only IGMC
    def __init__(self, chrs_graphs_temp, chrs_length_dict, segment_size, save_cell_id):
        super(CpGGraphDataset, self).__init__()
        self.chrs_graphs_temp = chrs_graphs_temp
        self.chrs_length_dict = chrs_length_dict
        self.link_num = chrs_length_dict[-1]
        self.segment_size = segment_size
        self.cell_num = chrs_graphs_temp[0].shape[1]
        self.half_seg = int((segment_size - 1)/2)
        #TODO original
        self.raw_x = F.one_hot(torch.cat((torch.arange(0, self.segment_size)+1,torch.zeros(self.cell_num))).long()).to(torch.float)
        #TODO without locus-aware
        # self.raw_x = F.one_hot(torch.cat((torch.ones(self.segment_size),torch.zeros(self.cell_num))).long()).to(torch.float)
        #TODO without any labels
        # self.raw_x = F.one_hot(torch.zeros(self.segment_size+self.cell_num).long()).to(torch.float)
        #TODO within cell-aware
        # self.raw_x = F.one_hot(torch.cat((torch.arange(0, self.segment_size)+self.cell_num,(torch.arange(0, self.cell_num)))).long()).to(torch.float)
        self.save_cell_id = save_cell_id

    def len(self):
        return self.link_num

    def get(self, index): 
        chr_ind = bisect.bisect_left(self.chrs_length_dict, index + 1) - 1
        chr_graph_csr = self.chrs_graphs_temp[chr_ind]
        rela_index = index - self.chrs_length_dict[chr_ind]
        col_ind = chr_graph_csr.indices[rela_index]
        row_ind = bisect.bisect_left(chr_graph_csr.indptr, rela_index + 1)
        window_start = (row_ind - 1) - self.half_seg
        window_end = row_ind + self.half_seg
        #CPG
        subgraph_csr = chr_graph_csr[window_start : window_end, :]
        # y
        y = torch.tensor(subgraph_csr[self.half_seg, col_ind] - 1).to(torch.bool)
        #remove target [row, col] info.
        subgraph_csr[self.half_seg, col_ind] = 1
        subgraph_coo = subgraph_csr.tocoo()
        chr_u = torch.from_numpy(subgraph_coo.row).to(torch.long)
        chr_v = torch.from_numpy(subgraph_coo.col + self.segment_size).to(torch.int16)#.to(torch.uint8)
        chr_r = torch.from_numpy(subgraph_coo.data - 1).to(torch.bool)
        #x
        x = self.raw_x[:]
        #edge_index
        edge_index = torch.stack([torch.cat([chr_u, chr_v]), torch.cat([chr_v, chr_u])], 0)
        #edge_type
        edge_type = torch.cat([chr_r, chr_r])
        if self.save_cell_id:
            subgrah_data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, chr=chr_ind, m=row_ind, n=col_ind)#, mat=subgraph_csr.astype(np.byte))
        else:
            subgrah_data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
        return subgrah_data

# Imputation
class CpGGraphImputationDataModule(pl.LightningDataModule):#*Only IGMC
    def __init__(self, y, segment_size=21, 
                 batch_size=2, n_workers=4, cell_nums=False):
        super().__init__()
        self.save_hyperparameters(ignore=['y'])
        self.y = y
        self.segment_size = segment_size
        self.half_seg = int((segment_size - 1)/2)
        self.batch_size = batch_size
        self.nw = n_workers
        if cell_nums:
            self.cell_num =cell_nums
        else:
            self.cell_num = y["chr1"].shape[1]
        self.save_cell_id = True
    
    def setup(self):
        time_start = time()
        graphs_temp, chrs_missing_length_dict, y_missing_temp = self.extract_imputation_subgraphs(self.cell_num)
        time_end = time()
        print("Running Time："+str(time_end - time_start)+"s")
        self.all = CpGGraphImputationDataset(graphs_temp, chrs_missing_length_dict, y_missing_temp, self.segment_size, self.save_cell_id)

    def extract_imputation_subgraphs(self, cell_nums):
        y_missing_temp = []
        chrs_graphs_temp = []
        chrs_missing_length_dict = [0]
        chr_missing_length = 0
        used_chrs=self.y.files
        for chr_name in tqdm(used_chrs):
            y_temp = self.y[chr_name][:, :cell_nums]
            y_missing = []
            select = list(set(np.argwhere(y_temp>-1)[:,0]))
            for s in select:
                missing = np.argwhere(y_temp[s,:]==-1)
                for i in missing:
                    if y_temp.shape[1] - len(missing) > 1:#atleast2
                        y_missing += [[s, i[0]]]
            y_missing = np.array(y_missing)
            y_missing_temp += [y_missing] 
            chr_missing_length += len(y_missing)
            chrs_missing_length_dict += [chr_missing_length]
            y_temp_add = np.full((self.half_seg, y_temp.shape[1]), -1)
            y_temp = np.concatenate((y_temp_add, y_temp, y_temp_add))
            y_temp_csr = ssp.csr_matrix(y_temp + 1)
            chrs_graphs_temp += [y_temp_csr]
        return chrs_graphs_temp, chrs_missing_length_dict, y_missing_temp


class CpGGraphImputationDataset(Dataset):#*Only IGMC
    def __init__(self, chrs_graphs_temp, chrs_missing_length_dict, y_missing_temp, segment_size, save_cell_id):
        super(CpGGraphImputationDataset, self).__init__()
        self.chrs_graphs_temp = chrs_graphs_temp
        self.cell_num = chrs_graphs_temp[0].shape[1]
        self.half_seg = int((segment_size - 1)/2)
        self.chrs_missing_length_dict = chrs_missing_length_dict
        self.y_missing_temp = y_missing_temp
        self.link_num = chrs_missing_length_dict[-1]
        self.segment_size = segment_size
        self.raw_x = F.one_hot(torch.cat((torch.arange(0, self.segment_size)+1,torch.zeros(self.cell_num))).long()).to(torch.float)
        self.save_cell_id = save_cell_id

    def len(self):
        return self.link_num

    def get(self, index): 
        chr_ind = bisect.bisect_left(self.chrs_missing_length_dict, index + 1) - 1
        chr_graph_csr = self.chrs_graphs_temp[chr_ind]
        y_missing = self.y_missing_temp[chr_ind]
        rela_index = index - self.chrs_missing_length_dict[chr_ind]
        row_ind, col_ind = y_missing[rela_index]
        row_ind = row_ind + self.half_seg + 1
        window_start = (row_ind - 1) - self.half_seg
        window_end = row_ind + self.half_seg
        #CPG
        subgraph_csr = chr_graph_csr[window_start : window_end, :]
        # y
        y = torch.tensor(subgraph_csr[self.half_seg, col_ind] - 1).to(torch.bool)
        #remove target [row, col] info.
        subgraph_csr[self.half_seg, col_ind] = 1
        subgraph_coo = subgraph_csr.tocoo()
        chr_u = torch.from_numpy(subgraph_coo.row).to(torch.long)
        chr_v = torch.from_numpy(subgraph_coo.col + self.segment_size).to(torch.int16)
        chr_r = torch.from_numpy(subgraph_coo.data - 1).to(torch.bool)
        #x
        x = self.raw_x[:]
        #edge_index
        edge_index = torch.stack([torch.cat([chr_u, chr_v]), torch.cat([chr_v, chr_u])], 0)
        #edge_type
        edge_type = torch.cat([chr_r, chr_r])

        if self.save_cell_id:
            subgrah_data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                                chr=chr_ind, m=row_ind - self.half_seg - 1, n=col_ind)
        else:
            subgrah_data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        return subgrah_data