import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, RGCNConv, global_add_pool
from torch_geometric.utils import dropout_adj
import pytorch_lightning as pl
from torchmetrics.functional import auroc
import numpy as np
from tabulate import tabulate

class GNN(pl.LightningModule):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(self, gconv=GCNConv, latent_dim=[32, 32, 32, 1], 
                 adj_dropout=0.2, force_undirected=False):
        super(GNN, self).__init__()
        self.adj_dropout = adj_dropout 
        self.force_undirected = force_undirected
        self.num_classes = 2#dataset.num_classes
        self.num_node_features = 4#dataset.num_node_features
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(self.num_node_features, latent_dim[0]))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1]))
        self.lin1 = Linear(sum(latent_dim), 128)

        self.lin2 = Linear(128, self.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class CpGGraph(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    def __init__(self, gconv=RGCNConv, latent_dim=[32, 64, 128, 128, 64, 32],#[32, 32, 32, 32], 
                 num_relations=2, num_bases=2, adj_dropout=0.2, 
                 force_undirected=False,
                 cell_num=25, segment_size=21 ,
                 multiply_by=1, lr=1e-3, lr_decay_factor=.90, warmup_steps=1000):
        super(CpGGraph, self).__init__(
            GCNConv, latent_dim, adj_dropout, force_undirected
        )
        self.segment_size = segment_size
        self.cell_num = cell_num
        self.sum_latent_dim = sum(latent_dim)
        self.save_hyperparameters()
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(self.segment_size + 1, latent_dim[0], num_relations, num_bases))


        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))

        self.CNN2 = nn.Sequential(nn.Conv2d(1, 4, (5, 5), stride=(1, 5), padding=(1, 2)), nn.ReLU(), nn.MaxPool2d(2, 2),
                                  nn.Conv2d(4, 1, (3, 3), stride=(1, 3), padding=(1, 1)), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.lin1_in_size = self.lin1_param(self.segment_size+self.cell_num, self.sum_latent_dim)
        self.lin1 = Linear(self.lin1_in_size, 40)

        self.hparams.lr = lr
        self.hparams.lr_decay_factor = lr_decay_factor
        self.hparams.warmup_steps = warmup_steps
        self.lin2 = nn.Linear(40, 1)


    def forward(self, data):
        (
            x, edge_index, edge_type
        ) = (
            data.x, data.edge_index, data.edge_type
        )

        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = torch.tanh(x)
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        x = torch.reshape(concat_states, (-1, self.segment_size+self.cell_num, self.sum_latent_dim)).unsqueeze(1)
        x = self.CNN2(x)
        x = torch.reshape(x.squeeze(), (-1, self.lin1_in_size))
        x = self.lin1(x)

        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)
        
        return self.lin2(x).squeeze(-1)
            

    def training_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = F.binary_cross_entropy_with_logits(y_hat, data.y.to(torch.float))
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            data = batch
            y_hat = self(data)
            y = data.y.to(torch.float)
            y_hat = y_hat
        return torch.stack((y_hat, y))
    
    def validation_epoch_end(self, validation_step_outputs):
        with torch.no_grad():
            validation_step_outputs = torch.cat(validation_step_outputs,1)
            y_hat = validation_step_outputs[0]
            y = validation_step_outputs[1]
            
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            self.log('val_loss', loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        y = data.y.to(torch.float)
        y_hat = y_hat
        return torch.stack((y_hat, y))

    def test_epoch_end(self, test_step_outputs):
        test_step_outputs = torch.cat(test_step_outputs, 1)
        y_hat = test_step_outputs[0]
        y = test_step_outputs[1]

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y = y.to(torch.int)
        y_hat = torch.sigmoid(y_hat)
        auroc_value = auroc(y_hat, y)
        self.log('loss', loss, sync_dist=True)
        self.log('AUROC', auroc_value, sync_dist=True)

        results_table = tabulate([['loss', np.round(loss.tolist(), 2)], 
                                ['AUROC', np.round(auroc_value.tolist()*100, 2)]
                                ], 
                                headers=['Metrics','Values'],
                                tablefmt='github')
        self.logger.experiment.add_text("GraphCpG Test Results", results_table)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        # update params
        optimizer.step(closure=optimizer_closure)
    
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer

    def lin1_param(self, H_in, W_in):
        #H
        H_out_1conv = math.floor((H_in + 2*1 - 1*(5-1) - 1)/1 + 1)
        H_out_1pool = math.floor((H_out_1conv + 2*0 - 1*(2-1) - 1)/2 + 1)
        H_out_2conv = math.floor((H_out_1pool + 2*1 - 1*(3-1) -1)/1 + 1)
        H_out_2pool = math.floor((H_out_2conv + 2*0 - 1*(2-1) - 1)/2 + 1)

        #W
        W_out_1conv = math.floor((W_in + 2*2 - 1*(5 - 1) - 1)/5 + 1)
        W_out_1pool = math.floor((W_out_1conv + 2*0 - 1*(2-1) - 1)/2 + 1)
        W_out_2conv = math.floor((W_out_1pool + 2*1 - 1*(3-1) - 1)/3 + 1)
        W_out_2pool = math.floor((W_out_2conv + 2*0 - 1*(2-1) - 1)/2 + 1)
        return H_out_2pool*W_out_2pool



