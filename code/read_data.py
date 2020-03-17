import os
import numpy as np
import torch
from sys import getsizeof

look_up = {"A": 0, "G": 1, "T": 2, "C": 3}


def DNA_to_onehot(sequence):
    max_seq_length = 142230 + 1
    data = np.zeros((4,max_seq_length))
    # print(data.shape)
    for index, letter in enumerate(sequence):
        data[look_up[letter.upper()]][index] = 1
    return torch.from_numpy(data).float()

def process_line(line, filepath):
    pieces = line.split(",")
    seq = pieces[1]
    chrom = pieces[0].split(":")[2]
    return seq, chrom


def importData(filepath):
    f = open(filepath)
    genes = {}
    label = None
    counter = 0
    for i, line in enumerate(f):
        line = line[:-1] #remove \n
        try:
            label = int(line.split(",")[-1])
        except:
            print("WARNING!!! File: ",filepath, " has no label assigned ... assuming label = 0." )
            label = 0
        seq, chrom = process_line(line, filepath)

        text_file = open("lengths_sequences.txt", "a")
        n = text_file.write(f"File: {filepath} \n")
        n = text_file.write(f"Current chrom: {chrom} with sequence length: {str(len(seq))} \n")
        n = text_file.write(f"--------")
        genes[chrom] = DNA_to_onehot(seq)
    text_file.close()
    return genes, label

def print_shapes(genes):
    print("-----")
    for gene, seq in genes.items():
        print(f"{gene} has shape {seq.shape} while holding {getsizeof(gene)} bytes")
    print("-----")

def obtain_all_files(filepath):
    file_list = []
    for filename in os.listdir(filepath):
        if filename.endswith(".txt"): 
            file_list.append(os.path.join(filepath, filename))
            continue
        else:
            continue
    return file_list

def set_up_tensors(genes):
    chroms = ['chr5','chr8', 'chr5','chr10', 'chr13','chr5', 'chr19']
    data_tensor = genes['chr1'] #removed chr1 from the whole set of chrs
    for chr in chroms:
        print("shape of current chromosome", chr.shape)
        print(data_tensor.shape)
        data_tensor = torch.cat((data_tensor, genes[chr]), 0)
    return data_tensor

def set_up_data(filepath):
    """
        Creates two dicts: {"filename": matrix}, {"filename": label} 
    """
    all_data = {}
    all_files = obtain_all_files(filepath)
    counter = 0
    for file in all_files:
        genes, label = importData(file)
        #data_tensor = set_up_tensors(genes)
        #print("shape of tensor per file: ", data_tensor.shape)
        #print_shapes(genes)
        #print(len(genes.items()))
   



"""
filepath_wgs0 = "C:\\Users\\laure\\OneDrive\\Desktop\\cnn-data\\wgs0.txt"
genes, label = importData(filepath_wgs0)
print_shapes(genes)


all_file_paths = obtain_all_files(filepath)

print(all_file_paths)"""

filepath = "C:\\Users\\laure\\OneDrive\\Desktop\\cnn-data"
set_up_data(filepath)
