# Written by P. Xydi, Feb 2022

######################################
# Import libraries
######################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

'''
Tools for preparing text data for sequence models
'''

def build_vocabulary(dataset):
    
    '''
    Builds the vocabulary from a provided dataset.
    
    INPUTS:
    - dataset:  list, list of samples
    
    OUTPUTS:
    - freq_sorted: dict, sorted token frequencies
    - w2idx: dict, dictionary mapping tokens to unique integers
    '''
    
    # Special tokens
    special_toks = ['__PAD__','__</e>__','__UNK__']

    freq = defaultdict(int)

    # Loop over all samples
    for i in range(len(dataset)):

        # Loop over all sentences in a sample
        for j in range(len(dataset[i])):

            # Loop over every (tok,tag) in a sentence
            for k in range(len(dataset[i][j])):

                # Get token
                tok = dataset[i][j][k][0].lower()

                # Update frequency values
                freq[tok] += 1

    # Sort dictionary in descending order
    freq_sorted = {k:v for k,v in sorted(freq.items(), 
                                         key=lambda item:item[1], reverse=True)}


    # Concatenate special tokens with vocabulary words (sorted)
    all_tokens  = list(freq_sorted.keys()) + special_toks

    print('Vocabulary size: {}\n'.format(len(all_tokens)))

    # Create word2index dictionary
    # Maps vocabulary words to a unique index 
    #(Note: from most frequent -> to least frequent)

    w2idx = {}

    for w in all_tokens:
        w2idx[w] = len(w2idx)

    return (freq_sorted, w2idx)


def pad_sentences(sample, vocab_dict, label_dict, seq_len=20, verbose = False):
    
    '''
    The pad_sentences function performs the following steps
    - maps tokens to vocabulary indexes
    - pads sequences according to seq_len
    
    INPUTS: 
    - sample         : list, list of (tok, tag) pairs for a sample sentence
    - vocab_dict     : dict, a word2index dictionary mapping tokens to unique integers
    - label_dict     : dict, a tag2index dictionary mapping named entity tags to unique integers
    - seq_len        : int, max length of sequences
    
    OUTPUT: 
    - sentence_tensor_pad : list, padded list of word indexes
    - tag_tensor_pad : list, padded list of label indexes
    
    Example: 
    --------
    Sample:        [['Rwandan', 'B-MISC'], ['refugee', 'O'], ['group', 'O'], ['calls', 'O'], 
                   ['for', 'O'], ['calm', 'O'], ['over', 'O'], ['census', 'O'], ['.', 'O']]

    Padded tensor:  [438, 4459, 328, 3772, 13, 4459, 96, 4459, 1, 4457, 4457, 4457, 4457, 4457, 4457]
    Padded labels:  [2, 3, 3, 3, 3, 3, 3, 3, 3, 4457, 4457, 4457, 4457, 4457, 4457]
    '''   
    
    # Get index for <UNK> token
    unknown_token='__UNK__'
    unk_ID = vocab_dict.get(unknown_token,0)  # unk_ID = 4459
    
    # Get list of tokens from sample (lowercased)
    tokens = [tok[0].lower() for tok in sample]
    if verbose:
        print('Tokens: \t{}'.format(tokens))
        
    # Get list of named entity tags from sample
    tags = [tok[1] for tok in sample]
    if verbose:
        print('Tags: \t\t{}\n'.format(tags))
    
    # Map tokens to vocab index
    sentence_tensor = []

    for tok in tokens:
        sentence_tensor.append(vocab_dict.get(tok,unk_ID))  # Replace OOV by unk_ID = 4459
    if verbose:
        print('Tensor:  \t{}'.format(sentence_tensor))
        
    # Map tags to integer index
    tag_tensor = [label_dict[tag] for tag in tags]
    if verbose:
        print('Label tensor: \t{}\n'.format(tag_tensor))
    
    ### PADDING
    # Pad sequences 
    len_0 = len(sentence_tensor)
    #print(len_0)

    if len_0 <= seq_len:
        pad_l = seq_len - len_0 # post-PADDING
        sentence_tensor_pad = sentence_tensor + [vocab_dict["__PAD__"]]*pad_l  # For padding tokens: PAD_ID = 4457
        tag_tensor_pad      = tag_tensor + [9]*pad_l       # For padding labels, I used 9
    else: 
        sentence_tensor_pad = sentence_tensor[0:seq_len]
        tag_tensor_pad      = tag_tensor[0:seq_len]
        

    return sentence_tensor_pad, tag_tensor_pad


def prep_data(dataset, seq_len, vocab, tag_map):

    X = []
    y = []

    # Loop over all samples
    for i in range(len(dataset)):
        # Loop over all sentences in a sample
        for j in range(len(dataset[i])):

            sample = dataset[i][j]

            sent_tensor, label_tensor = pad_sentences(sample, vocab, tag_map, seq_len, verbose=False)

            X.append(sent_tensor)
            y.append(label_tensor)
            
    return np.array(X), np.array(y)          