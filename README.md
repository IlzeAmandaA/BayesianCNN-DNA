# Bayesian CNNs and DNA
 Temporariry Github repository for the DNA project

# How the current files work
- file main.py is the main file to run. Ideally this file is run from the terminal line, as: python main.py
  Within this file you have to specify the filepath of the input data (filepath=' ').
  The expect data format is .txt, where there is 1 file per person, each gene on a new line, where the contents
  are seperated by a comma: seq_id,sequence,label
  In the main file you can also change the percentage of the data used for validation as well as the number of 
  epochs the model is going to be trained on (you can find this as the hyper-parameters of the model)
  

- file modelV2.py containts the gated-attention network. Within this file you can find the dimensionalities of 
 the network, as well as its structure


- file read_data.py accesses the txt files, in this file you must specify the maximum length of the 
  genes as the padding will be performed accordingly (MAX_LEGNTH parameter)
  
- all results are saved in the folder 'output', in particular, during training the model is saved after every 10 epochs
  as well as the corresponding loss obtained, and in a txt file the performance on the left-out test set is reported

