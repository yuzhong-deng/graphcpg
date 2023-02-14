# Data format
Though GraphCpG only used the methylation matrix for imputation, to compare with other models fairly, the genome and the location files are all prepared.
We continue to use the form of input file by Gdewael https://github.com/gdewael/cpg-transformer/tree/main/data
* Methylation matrix ```y```
* Genome ```X```
* Location ```pos```
# Datasets
* HCC, MBL, and Hemato
  
  Same as https://github.com/gdewael/cpg-transformer/tree/main/data

* Neuron-Mouse

Enter the Neuron-Mouse folder (change the ```output_directory``` in  ```downloadMouseLuo.py``` to your own download folder)
```
python downloadMouseLuo.py
```


  

* Neuron-Homo

Enter the Neuron-Homo folder, download GSM file using links from the excel table.
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
