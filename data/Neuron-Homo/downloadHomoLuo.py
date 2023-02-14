import pandas as pd
import numpy as np
import re
import tarfile
import wget
from tqdm import tqdm
import os
import time
excel_df = pd.read_excel("Luo_homo_download_titleANDlinks.xlsx", index_col=None, header=None)
patternRandomPrimerIndex = r"(.*)AD008(.*)"
link_list = []
for id,title in enumerate(excel_df.iloc[0,:]):
    if re.match(patternRandomPrimerIndex, title)!=None:
        link_list.append(excel_df.iloc[1,id])
output_directory = "E:/Homo"
for link in tqdm(link_list[:]):
    download_link = 'https:' + link[4:]
    download_flag = True
    while download_flag:
        try:
            filename = wget.download(download_link, out=output_directory)
            # print(filename)
            download_flag = False
        except:
            print("Retry...")
            time.sleep(1)