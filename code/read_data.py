import os
import numpy as np
import torch
from sys import getsizeof

look_up = {"A": 0, "G": 1, "T": 2, "C": 3}
MAX_LENGHT = 142230 #change depending on data


""" NOTE:
     The function set_up_tensors
"""

def DNA_to_onehot(sequence):
    """
        Creates a matrix of size (4, max_seq_length) with only 0 vectors then fills it up accordingly.
    """
    max_seq_length = MAX_LENGHT + 1
    data = np.zeros((4,max_seq_length))
    # print(data.shape)
    for index, letter in enumerate(sequence):
        data[look_up[letter.upper()]][index] = 1
    return torch.from_numpy(data).float()

def process_line(line, filepath):
    """
        Processes a given line to aquire the sequence and the name of the chromosome
    """
    pieces = line.split(",")
    seq = pieces[1]
    #chrom = pieces[0].split(":")[2]
    chrom = pieces[0]
    return seq, chrom


def importData(filepath, to_write = False):
    """
        Imports the data and prints if needed. Also provides a warning if we miss a label.
    """

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


        if to_write:
            text_file = open('output/'+"lengths_sequences.txt", "a")
            text_file.write("File: {} \n".format(filepath))
            text_file.write("Current chrom: {} with sequence length: {} \n".format(chrom, len(seq)))
            text_file.write("--------")

        genes[chrom] = DNA_to_onehot(seq)
        # print('crom {} label {}'.format(chrom, label))
    if to_write:
        text_file.close()
        # print(f'min length = {min_len} and max_length = {max_length}')
    # print('final label {}'.format(label))
    return genes, label

def print_shapes(genes):
    print("-----")
    for gene, seq in genes.items():
        print("{} has shape {} while holding {} bytes".format(gene, seq.shape, getsizeof(gene)))
    print("-----")

def obtain_all_files(filepath):
    """
        Returns all files for a specific folder that end in .txt
    """
    file_list = []
    for filename in os.listdir(filepath):
        if filename.endswith(".txt"): 
            file_list.append(os.path.join(filepath, filename))
            continue
        else:
            continue
    return file_list

def set_up_tensors(genes, chroms):
    """
        Given a dict of genes, stack them and return them in one big tensor (x_genes, n_embedding, n_sequence)
    """
    # chroms = ['chr5','chr8', 'chr9','chr10','chr13','chr15','chr19']
    chr1 = chroms[0]
    data_tensor = genes[chr1].unsqueeze(dim=0) #removed chr1 from the whole set of chrs
    for chrm in chroms[1:]:
        data_tensor = torch.cat(([data_tensor, genes[chrm].unsqueeze(dim=0)]), dim = 0)
    return data_tensor

def convert_labels_scalar_to_vector(labels):
    all_labels = np.zeros((len(labels), 2))
    for index, label in enumerate(labels):
        all_labels[index][label] = 1
    return torch.from_numpy(all_labels).float()

def set_up_data(filepath, n, print_boo = False):
    """
        Creates two dicts: {"filename": matrix}, {"filename": label} 
    """
    all_data = {}
    all_files = obtain_all_files(filepath)
    first_iteration = True
    gene_ids=None
    labels = []

    counter = 0
    if print_boo:
        print("==== Loading data ====")

    for idx_file,file in enumerate(all_files):

        if print_boo:
            print("Current file: {}".format(file))

        if counter == n:
            break
        counter += 1
        genes, label = importData(file, to_write = False)

        # set up the gene order according to the first file read
        if idx_file==0:
            gene_ids = list(genes.keys())

        labels.append(label)
        if first_iteration:
            data_tensor = set_up_tensors(genes, gene_ids).unsqueeze(dim = 0)
            first_iteration = False
        else:
            data = set_up_tensors(genes, gene_ids).unsqueeze(dim = 0)
            data_tensor = torch.cat(([data_tensor, data]), dim = 0)

        #print(data_tensor.shape)
    if print_boo:
        print("==== Finished loading data ====")
    # print(labels)
    # label_tensor = convert_labels_scalar_to_vector(labels)
    label_tensor = torch.tensor(labels).view(-1,1)
    return data_tensor, label_tensor




