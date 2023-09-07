import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import requests
import gzip
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import bisect

from collections import OrderedDict
from scipy import stats

df = np.load("") # methylation matrix in dataframe
used_subpop = [] # labels list

new_df = pd.DataFrame(np.zeros((122,122)))
for i,cell in enumerate(tqdm(df.columns)):
    for j,other_cells in enumerate(df.columns):
        if new_df.iloc[j,i] == 0:
            new_df.iloc[i,j] = df[(df[[cell,other_cells]]>-1).sum(axis=1)==2][[cell,other_cells]].corr(method="spearman").iloc[0,1]
            new_df.iloc[j,i] = new_df.iloc[i,j]
        else:
            pass

new_df.columns = df.columns#new_col_names_columns
new_df.index = df.index#new_col_names_columns

# Create a categorical palette to identify the networks
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_subpop), network_pal))
# Convert the palette to vectors that will be drawn on the side of the matrix
networks = new_df.columns.get_level_values("Subpopulation")
network_colors = pd.Series(networks, index=new_df.columns).map(network_lut)

g_c = sns.clustermap(new_df, center=0, cmap="vlag", col_colors=network_colors, #col_colors=network_colors,df1_corr
               dendrogram_ratio=.2,
               colors_ratio=.2,
               xticklabels=True, yticklabels=False,
            #    col_cluster=False,
               cbar_kws={'orientation':'horizontal'},
               cbar_pos=(.85, .95, .05, .04),
               linewidths=.0, figsize=(20, 4))
g_c.ax_row_dendrogram.remove()
g_c.ax_col_dendrogram.set_title('Imputed Subpopulation Dendrogram')
g_c.ax_col_dendrogram.set_xlabel("")
plt.setp(g_c.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=10)
