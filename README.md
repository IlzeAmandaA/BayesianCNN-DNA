## Multiple Instance Learning for Relevant Gene Identification

by Ilze Amanda Auzina (ilze.amanda.auzina@gmail.com) and Laurence Bont (laurencebont@gmail.com)

# Overview
Pytorch implementation of Deep Multiple Instance Learning for Genomic sequence data

# Installation 
Tested with python 3.5.2 with PyTorch 1.4 (CUDA 10.1), should work without cuda as well. 

# Abstract
Many human diseases or health conditions have an underlying genetic component. In order to dis- cover them the standard procedure is to proceed from gene level up, which requires detailed gene annotation as a starting point. However, gene annotation is a very difficult task, which is not readily available. Therefore, we propose to re- define the problem via multiple instance learning (MIL) setting, which allows to use a person level label, thus avoiding the need for detailed gene annotation. We show the first proof-of-concept implementation on a simulation data set consist- ing of only 160 people. The novel application succeeds to identify the relevant genes despite the small data set size and the high variance in gene lengths utilized.

For a complete explanation of the model and the findings, please read the `report.pdf`.

# Model
The core component of the model is a shallow MC-dropout CNN. Hence, we consider the DNA as a 1D sequence with 4 channels (ACGT). Furhtermore, the CNN is set as part for Multiple Instance Learning (MIL), where instead of having a single instance for a target variable, there is a bag of instances X = { x1, ..., xk } that are independent of each other. 

# How to Use

root directory
- train.py : Trains a shallow MC-dropout CNN with the Adam optimization. 
- test.py : Returns the attention weights for the test set (see whether the sequence was identified) 
  
code directory
- model.py: Containts the gated-attention network. Within this file you can find the dimensionalities of 
 the network, as well as its structure. The loss function is defined as corss-entropy loss (binary classification problem). 

- read_data.py: accesses the txt files, in this file you must specify the maximum length of the genes as the padding will be performed accordingly (MAX_LEGNTH parameter), as well as the data order (DATA_ORDER parameter) to keep the data loading order correct across participants
  
# Data
To replicate the 'fake' data used in the paper (section 4.1.1), please visit [repo](https://github.com/IlzeAmandaA/simDNAgen)

