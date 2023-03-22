# Data format
Though GraphCpG only used the methylation matrix for imputation, to compare with other models fairly, the genome and the location files are all prepared.
We continue to use the form of input file by Gdewael (https://github.com/gdewael/cpg-transformer/tree/main/data#dataset-processing-instructions)
* Methylation matrix ```y```
* Genome ```X```
* Location ```pos```
# Datasets
* HCC, MBL, and Hemato
  
  Same as [HCC](https://github.com/gdewael/cpg-transformer/tree/main/data#hcc-dataset), [MBL](https://github.com/gdewael/cpg-transformer/tree/main/data#mbl-dataset), and [Hemato](https://github.com/gdewael/cpg-transformer/tree/main/data#hemato-dataset)

* Neuron-Mouse (Luo_Mouse)

Enter the Neuron-Mouse folder (change the ```output_directory``` in  ```downloadMouseLuo.py``` to your own download folder)
```
python downloadMouseLuo.py
```


  

* Neuron-Homo (Luo_Homo)
  
  **Methylation matrix and positions**

  Enter the Neuron-Homo folder, download GSM files using official links from the excel table.
  ```
  python downloadHomoLuo.py
  ```
  Unzip the packages
  ```
  python unzipHomoLuo.py
  ```
  Untar the packages
  ```
  python untarHomoLuo.py
  ```
  Encode ```y``` and ```pos```
  ```
  sh encodeHomoLuo.sh
  ```
  Combine encoded ```y``` and ```pos```
  ```
  sh combineHomoLuo.sh
  ```
  **Genome**
  
  (hg38) Same as [Hemato](https://github.com/gdewael/cpg-transformer/tree/main/data#genome-3) 
  
# Genomic Contexts
- notebook
