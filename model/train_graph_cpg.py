import numpy as np
from argparse import ArgumentParser
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.profiler import PyTorchProfiler
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate
from time import time
import json

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass

parser = ArgumentParser(description='Training script for CpG', formatter_class=CustomFormatter)

dm_parse = parser.add_argument_group('DataModule', 'Data Module arguments')
dm_parse.add_argument('--dataset_path', type=str, default="",
                      help='Datapath of datasets')
dm_parse.add_argument('--dataset', type=str, default="HCC",
                      help='name of datasets')
dm_parse.add_argument('--segment_size', type=int, default=21,
                      help='Bin size in number of CpG sites (columns) that every batch will contain. If GPU memory is exceeded, this option can be lowered.')
dm_parse.add_argument('--val_keys', type=str, nargs='+', default=['chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19'],
                      help='Names/keys of validation chromosomes.')
dm_parse.add_argument('--test_keys', type=str, nargs='+', default=['chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12'], 
                      help='Names/keys of test chromosomes.')
dm_parse.add_argument('--batch_size', type=int, default=10240,
                      help='Batch size.')
dm_parse.add_argument('--n_workers', type=int, default=12,
                      help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')
                    
model_parse = parser.add_argument_group('Model', 'CpGTIGMC Hyperparameters')
model_parse.add_argument('--gpu', type=int, default=0,
                         help='id of used GPU.')
model_parse.add_argument('--latent_dim', type=list, default=[32, 64, 128, 128, 64, 32],
                         help='latent_dim.')
model_parse.add_argument('--adj_dropout', type=float, default=0.2,
                         help='args.adj_dropout')
model_parse.add_argument('--lr', type=float, default=1e-3,
                         help='Learning rate.')
model_parse.add_argument('--lr_decay_factor', type=float, default=.99,#.5,
                         help='Learning rate multiplicative decay applied after every epoch.')
model_parse.add_argument('--warmup_steps', type=int, default=100,
                         help='Number of steps over which the learning rate will linearly warm up.')
model_parse.add_argument('--maxepochs', type=float, default=10,#type=int, default=1,
                         help='Number of maximum epoch.')

log_parse = parser.add_argument_group('Logging', 'Logging arguments')
log_parse.add_argument('--tensorboard', type=bool, default=True,
                       help='Whether to use tensorboard. If True, then training progress can be followed by using (1) `tensorboard --logdir logfolder/` in a separate terminal and (2) accessing at localhost:6006.')
log_parse.add_argument('--log_folder', type=str, default='logfolder',
                       help='Folder where the tensorboard logs will be saved. Will additinally contain saved model checkpoints.')
log_parse.add_argument('--experiment_name', type=str, default='graph_cpg',
                       help='Name of the run within the log folder.')
log_parse.add_argument('--earlystop', type=bool, default=True,
                       help='Whether to use early stopping after the validation loss has not decreased for `patience` epochs.')
log_parse.add_argument('--patience', type=int, default=5,
                       help='Number of epochs to wait for a possible decrease in validation loss before early stopping.')

if __name__ == '__main__':
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    data_path = args.dataset_path
    datset_dict = {"HCC":"y_HCC.npz"}#, 
                    # "MBL":"y_MBL.npz",
                    # "Hemato":"y_Hemato.npz",
                    # "Luo_Mouse":"y_Luo_Mouse.npz",#"Neuron-Mouse"
                    # "Luo_Homo":"y_Luo_Homo.npz"#"Neuron-Homo"
                    # }

    y = np.load(data_path + datset_dict[args.dataset])
    from datamodules import CpGGraphDataModule
    from graphcpg import CpGGraph


    datamodule = CpGGraphDataModule(y, segment_size=args.segment_size,
                                val_keys=args.val_keys, test_keys=args.test_keys, 
                                batch_size=args.batch_size, n_workers=args.n_workers,
                                cell_nums=False)

    model = CpGGraph(latent_dim=args.latent_dim, adj_dropout=args.adj_dropout, cell_num=datamodule.cell_num,
                    segment_size=args.segment_size,
                    lr=args.lr, lr_decay_factor=args.lr_decay_factor, warmup_steps=args.warmup_steps)

    maxepochs = args.maxepochs


    args.log_folder += "/formal/"+str(maxepochs)+"/" + args.dataset
    if maxepochs < 1:
        limit_batch = maxepochs
        limit_batch_test = 1.0
        maxepochs = 1
    else:
        limit_batch = 1.0
        limit_batch_test = 1.0
        maxepochs = int(maxepochs)

    callbacks = [ModelCheckpoint(monitor='val_loss', mode='min',
                                # every_n_train_steps=save_steps,
                                save_top_k=1, save_last=True), 
                TQDMProgressBar(refresh_rate=1)]

    if args.tensorboard:
        logger = TensorBoardLogger(args.log_folder, name=args.experiment_name)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks += [lr_monitor]

    #wait for graph cards
    gpu_id = args.gpu
    time_start = time()
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, gpus=[gpu_id], 
                                        max_epochs = maxepochs,
                                        limit_train_batches=limit_batch,
                                        limit_val_batches=limit_batch,
                                        limit_test_batches=limit_batch_test)
    datamodule.setup()
    time_process = time()
    process_timer = time_process - time_start
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    time_train = time()
    train_timer = time_train - time_process
    # automatically loads the best weights
    trainer.test(model, datamodule.test_dataloader(), ckpt_path='best')
    time_test = time()
    test_timer = time_test - time_train

    writer = SummaryWriter(log_dir = logger.log_dir + '/table')
    print("Total preprocess time:"+str(np.round(process_timer/60, 2))+"mins")
    print("Total train time:"+str(np.round(train_timer/60, 2))+"mins")
    print("Total test time:"+str(np.round(test_timer/60, 2))+"mins")

    time_table = [["time"],
                  ["preprocess", str(np.round(process_timer/60, 2))], 
                  ["train", str(np.round(train_timer/60, 2))], 
                  ["test", str(np.round(test_timer/60, 2))]]
    time_table_header = ["time"]
    time_table_tabulate = tabulate(time_table, headers="firstrow", tablefmt='github')
    writer.add_text("GraphCpG Time (mins)", time_table_tabulate)
    writer.close()

    with open(logger.log_dir+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    

