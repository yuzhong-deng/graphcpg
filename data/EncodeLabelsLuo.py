import argparse
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Encode CpG labels from tsv file into compact format')

parser.add_argument('dataFile', type=str, metavar='<folder name>',
                    help='data file')
parser.add_argument('EncodedGenome', type=str, metavar='<.npz file>',
                    help='Encoded genome from EncodeGenome.py.')
parser.add_argument('y_outFile', type=str, metavar='<.npz file>',
                    help='output file to save encoded labels in.')
parser.add_argument('pos_outFile', type=str, metavar='<.npz file>',
                    help='output file to save encoded positions of labels in.')
parser.add_argument('--chroms', nargs="+", type=str, required=True,
                    help='ordering of chromosomes in the fasta file')
parser.add_argument('--prepend_chr', action='store_true',
                    help='whether to prepend the str "chr" to the names of chromosomes given in --chroms.')

args = parser.parse_args()

chroms = args.chroms
if args.prepend_chr:
    # chroms = ["chr" + c for c in chroms]
    chroms = [c for c in chroms]

# path = "/home/dyz/repo/cpg-transformer-main/data/HCC"
# path = 'E:/Homo'
path = os.getcwd()
os.chdir(path)
# print(os.getcwd())

# print('Reading data ...')
cell_data_folder_name = args.dataFile

X_encoded = np.load(args.EncodedGenome)

y_encoded = {}
pos_encoded = {}
for chrom_short_name in tqdm(chroms):
    chrom_name = "chr" + chrom_short_name
    # print('Encoding',chrom_name,'...')

    tsv_path = path+'/'+args.dataFile+'/'+"allc_"+args.dataFile[:-7]+chrom_short_name+".tsv"
    dat = pd.read_csv(tsv_path, sep='\t', header=0, dtype={0:'string', 2:'string',3:'string'})
    datsubset = dat[dat["strand"]=="+"]
    # dat = dat[dat["mc_class"]=="CGG"]
    datsubset = datsubset[datsubset.mc_class.str.contains(r"CG.")]

    X_chrom = X_encoded[chrom_name]
    indices = np.where(X_chrom==2)[0]#seq上C开头的位置
    
    label_chrom = datsubset["methylated"].values.astype('int8')
    
    subset_ind_C = np.in1d(datsubset.iloc[:,1]-1, indices)
    
    y_encoded[chrom_name] = label_chrom[subset_ind_C].astype('int8')
    pos_encoded[chrom_name] = (datsubset.iloc[:,1]-1)[subset_ind_C].astype('int32')

np.savez_compressed(args.y_outFile, **y_encoded)
np.savez_compressed(args.pos_outFile, **pos_encoded)
