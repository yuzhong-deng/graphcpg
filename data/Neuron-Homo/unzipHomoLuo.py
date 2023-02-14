from tqdm import tqdm
import os
import gzip
import shutil

output_directory = "E:/Homo"
# Un-gzip
print("un_gzip:")
cells_folder_list = []
for folder in tqdm(os.listdir(output_directory)[:]):
    # print(folder)
    folder_path = output_directory+'/'+folder
    for file in os.listdir(folder_path):
        file_path = folder_path+'/'+file
        with gzip.GzipFile(file_path, 'rb') as f_in:
            with open(file_path.replace(".gz",""), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
                f_in.close()
                f_out.close()
        os.remove(file_path)