## Classification

Two training files that contain positive and negative samples respectively are needed.

Usage: python deepboost.py source_dir dataset_name test_file result_file, where

source_dir: directory where two training files are stored

dataset_name: name of the dataset and two training files should have the name of dataset_name+".positives.fa" and dataset_name+".negatives.fa"

test_file: the name of the test file

result_file: the file in which the prediction result on the test file will be saved

Example:
```
python deepboost.py ./data/ ALKBH5_Baltz2012.train ./data/ALKBH5_Baltz2012.ls.positives.fa ./result/pred.txt
```
----------------------------------------------------------------------------------------------
## Motif Generation

Two training files that contain positive and negative samples respectively are needed.

Usage: python motif.py source_dir dataset_name, where

source_dir: directory where two training files are stored

dataset_name: name of the dataset and two training files should have the name of dataset_name+".positives.fa" and dataset_name+".negatives.fa"

Note:
It accepts a file "dict_motif.fa", which currently consists of all 8-mers. For generating motifs of length n, this file should be replaced to contain all n-mers.
It will generate a temporary file called "tmp.fa" and the motif result will be stored at dataset_name+".motif.fa"

Example:
```
python motif.py ./data/ ALKBH5_Baltz2012.train ./data/dict_motif.fa ./result/
```
----------------------------------------------------------------------------------------------
##Regression

One training file that contains RNA sequences and their corresponding affinity values is needed.

Usage: python RNAcompete.py filename_train filename_test output_dir, where

filename_train: filename of the training data

filename_test: filename of the testing data

output_dir: output directory

Note:
The predicted affinity for the test data and a scatter plot with x-axis representing the normalized ground truth and y-axis representing the predicted affinity on the test data will be stored under the output directory

Example:
```
python RNAcompete.py ./data/RNAcompete_7mer_HNRNPC.train.txt ./data/RNAcompete_7mer_HNRNPC.test.txt ./result/ RNAcompete_7mer_HNRNPC
```