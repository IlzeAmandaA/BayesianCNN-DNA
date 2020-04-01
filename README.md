## Multiple Instance Learning for Relevant Gene Identification

by Ilze Amanda Auzina (ilze.amanda.auzina@gmail.com) and Laurence Bont (laurencebont@gmail.com)

#Overview
Pytorch implementation of Deep Multiple Instance Learning for Genomic sequence data

#Installation 
Tested with python 3.5.2 with PyTorch 1.4 (CUDA 10.1), should work without cuda as well. 

#Content 
The code can be run to identify relevant genes based on a subject-level label. At the moment simulation data is used stored in the folder data_simulation. If real life data is available, you can store it in folder data, and change the corresponding file path in the code. Important to note that the expected input file format is a .txt file, where each gene is stored on a new line and the data is organized as follows: geneid,sequnece,label

#How to Use
- main.py : Trains a shallow MC-dropout CNN with the Adam optimization. The best model is identified based on the loss obtained on the validation set and stored in the output folder. Furthermore, the loss and error obtained during training are also stored in the output folder for both validation and training set. Intermediate models that are saved during training are stored in log folder to allow to resume training at a later point. Lastly, the bags labels and instance labels are printed during the training process. 
  
- modelV2.py: Containts the gated-attention network. Within this file you can find the dimensionalities of 
 the network, as well as its structure. The loss function is defined as corss-entropy loss (binary classification problem). 

- read_data.py: accesses the txt files, in this file you must specify the maximum length of the genes as the padding will be performed accordingly (MAX_LEGNTH parameter), as well as the data order (DATA_ORDER parameter) to keep the data loading order correct across participants
  

