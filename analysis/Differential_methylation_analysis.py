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

df = pd.read_excel('mmc5.xlsx') #supplement file form "Farlik, Matthias, Florian Halbritter, Fabian Müller, Fizzah A. Choudry, Peter Ebert, Johanna Klughammer, Samantha Farrow, et al. “DNA Methylation Dynamics of Human Hematopoietic Stem Cell Differentiation.” Cell Stem Cell 19, no. 6 (December 1, 2016): 808–22. https://doi.org/10.1016/j.stem.2016.10.019."
select_col = df.columns[6:229]
df_official = df[select_col]

df_imputed = pd.DataFrame(index=df.index, columns=df_y_subpops_imputed.columns)
for chr_start_end in tqdm(df[["Chromosome","Start","End"]].iterrows(), total=df.shape[0]):
    chr, start, end = chr_start_end[1]
    # print('chr, start, end: ', chr, start, end)
    df_dml_imputed = df_y_subpops_imputed.loc[(chr),:]
    df_dml_imputed = df_dml_imputed[(df_dml_imputed.index < end) & (df_dml_imputed.index > start)]
    for cell in df_dml_imputed.columns:
        y_cell_region = df_dml_imputed.loc[:,([cell])]
        if y_cell_region.empty:
            df_imputed.loc[chr_start_end[0],cell] = np.nan
        else:
            y_cell_region = (y_cell_region).astype(np.float64)
            y_cell_region = y_cell_region[y_cell_region>-1].dropna().mean()
            df_imputed.loc[[chr_start_end[0]],cell] = y_cell_region.values

df_raw = pd.DataFrame(index=df.index, columns=df_y_subpops_raw.columns)
for chr_start_end in tqdm(df[["Chromosome","Start","End"]].iterrows(), total=df.shape[0]):
    chr, start, end = chr_start_end[1]
    # print('chr, start, end: ', chr, start, end)
    df_dml_raw = df_y_subpops_raw.loc[(chr),:]
    df_dml_raw = df_dml_raw[(df_dml_raw.index < end) & (df_dml_raw.index > start)]
    for cell in df_dml_raw.columns:
        y_cell_region_raw = df_dml_raw.loc[:,([cell])]
        if y_cell_region_raw.empty:
            df_raw.loc[chr_start_end[0],cell] = np.nan
        else:
            y_cell_region_raw = (y_cell_region_raw).astype(np.float64)
            y_cell_region_raw = y_cell_region_raw[y_cell_region_raw>-1].dropna().mean()
            df_raw.loc[[chr_start_end[0]],cell] = y_cell_region_raw.values

sns.set_theme('paper')
# Imputed & Raw
sub_types_names = df_y_subpops_raw.columns.get_level_values(0).values
df_raw_RR = pd.DataFrame({'Methylation levels':df_raw.mean().values, "Subpopulations":sub_types_names, "Data": ["Raw"]*122})
df_imputed_RR = pd.DataFrame({'Methylation levels':df_imputed.mean().values, "Subpopulations":sub_types_names, "Data": ["Imputed"]*122})

# Official
select_sub_col = []
select_sub_columns = []
for subpop in df_official.columns:
    if subpop[:3] in used_subpop:
        select_sub_col.append(subpop)
        select_sub_columns.append(subpop[:3])
    elif subpop[:4] in used_subpop:
        select_sub_col.append(subpop)
        select_sub_columns.append(subpop[:4])
    else:
        pass

df_official_RR = pd.DataFrame({'Methylation levels':df_official[select_sub_col].mean().values, "Subpopulations":select_sub_columns, "Data": ["Bulk"]*191})
df_RR_total = pd.concat([df_raw_RR, df_imputed_RR, df_official_RR], axis=0)
ax = sns.boxplot(data=df_RR_total, x="Subpopulations", y="Methylation levels", hue="Data", order=order_subpop)
ax = sns.swarmplot(data=df_RR_total.dropna(), x="Subpopulations", y="Methylation levels", hue="Data", size=4, edgecolor="white", linewidth=.5, order=order_subpop, alpha=0.6)

# Add significance asterisk bar
x1, x2 = 2.5, 4.5   # x coordinates of the groups to compare
x11, x12, x21, x22 = 1.6, 3.4, 3.6, 5.4
y, h, col = .85, .02, 'k' # y coordinate, height and color of the bar
y_text = .83
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col) # plot the bar
plt.plot([x11, x12], [y, y], lw=1.5, c=col) # plot the 2 bar
plt.plot([x21, x22], [y, y], lw=1.5, c=col) # plot the 3 bar

A_seq = df_RR_total.dropna()[df_RR_total.dropna()["Subpopulations"]=="CMP"]["Methylation levels"]
B_seq = df_RR_total.dropna()[df_RR_total.dropna()["Subpopulations"]=="CLP"]["Methylation levels"]
if len(A_seq) != len(B_seq):
    cut = np.min([len(A_seq), len(B_seq)])
    A_seq = A_seq[:cut]
    B_seq = B_seq[:cut]
stat, p = stats.wilcoxon(x=A_seq, y=B_seq, alternative='two-sided')
if p < 0.001:
    sig_symbol = '***'
elif p < 0.01:
    sig_symbol = '**'
elif p < 0.05:
    sig_symbol = '*'
else:
    sig_symbol = 'n.s.'

add_3 = 0

for i in range(18):
    count = [18, 18, 2460, 18, 18, 4640, 19, 19, 3540, 22, 22, 3540, 21, 21, 3540, 24, 24, 3530]
    if i % 3 == 0:
        add_3 = add_3 + 0.12
    ax.text(-0.6 + 0.3*i + add_3, .55, f' {count[i]}\n cells', size = 8)#, ha='center', va='bottom')

plt.text((x1+x2)*.5, y+h, sig_symbol, ha='center', va='bottom', color=col) # plot the asterisk
plt.text(x1, y_text, "Myeloid progenitors", ha='center', va='bottom', color=col) # plot the asterisk
plt.text(x2, y_text, "Lymphoid progenitors", ha='center', va='bottom', color=col) # plot the asterisk
plt.title("Differentially methylated regulatory regions")
plt.ylabel("Average methylation level by cells")


plt.tight_layout()
plt.legend(loc="center right", bbox_to_anchor=(1.2, 0.5))
