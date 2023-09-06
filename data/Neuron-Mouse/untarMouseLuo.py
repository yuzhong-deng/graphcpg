from tqdm import tqdm
import os
import tarfile

output_directory = ""
# Un-tar
print("un_tar:")
for file in tqdm(os.listdir(output_directory)):
    # print(output_directory+'/'+file)
    tar_filename = output_directory+'/'+file
    tar = tarfile.open(tar_filename)
    tar.extractall(output_directory)
    tar.close()
    os.remove(tar_filename)
