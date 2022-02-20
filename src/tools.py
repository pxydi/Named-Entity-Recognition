# Written by P. Xydi, Feb 2022

######################################
# Import libraries
######################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.cm as cm
color_1 = cm.get_cmap("Set2")(2) # set blue 
color_2 = cm.get_cmap("Set2")(1) # set orange
from sklearn.metrics import ConfusionMatrixDisplay

######################################
def sentence_distribution(dataset, label = 'training', to_plot = False):
    '''
    Plots the distribution of sentences in a given dataset
    
    INPUTS:
    - dataset:  list, list of samples
    - label:    str, used in the title of the output plot

    OUTPUT:
    - histogram of the distribution of sentences in dataset
    - nbr_sents : list, number of sentences per sample in dataset
    
    '''
######################################

    # Create empty list
    nbr_sents = []

    for i in range(len(dataset)):
        nbr_sents.append(len(dataset[i]))
        
    if to_plot:
        
        # Plot the sentence distibution

        # Barplot and font specifications
        barplot_specs = {"color": color_1, "alpha": 0.7, "edgecolor": "grey"}
        label_specs   = {"fontsize": 12}
        title_specs   = {"fontsize": 14, "fontweight": "bold", "y": 1.03}

        plt.figure(figsize=(8,4))

        plt.hist(nbr_sents, bins = 20, **barplot_specs)
        plt.xlabel('Nbr of sentences per sample', **label_specs)
        plt.ylabel('Nbr of samples',**label_specs)
        plt.title('Distribution of sentences in {} set'.format(label),**title_specs)
        plt.show()

    return np.sum(nbr_sents)

######################################
def plot_token_distribution(dataset, label):
    '''
    Plots the distribution of tokens in the sentences of a 
    given dataset.
    
    INPUTS:
    - dataset:  list, list of samples
    - label:    str, used in the title of the output plot

    OUTPUT:
    - histogram of the distribution of tokens in dataset
    - nbr_tokens: list, number of tokens per sentence in dataset
    
    '''
######################################

    # Create empty list
    nbr_tokens = []
    
    # Count tokens in sentences and append to list
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            nbr_tokens.append(len(dataset[i][j]))
            
            
    # Plot the sentence distibution
    
    # Barplot and font specifications
    barplot_specs = {"color": "mediumpurple", "alpha": 0.7, "edgecolor": "grey"}
    label_specs   = {"fontsize": 12}
    title_specs   = {"fontsize": 14, "fontweight": "bold", "y": 1.03}

    plt.figure(figsize=(8,4))

    plt.hist(nbr_tokens, bins = 20, **barplot_specs)
    plt.xlabel('Nbr of tokens per sentence', **label_specs)
    plt.ylabel('Nbr of sentences',**label_specs)
    plt.title('Distribution of tokens in {} set'.format(label),**title_specs)
    plt.show()
    
    return nbr_tokens

######################################
def target_sample_distribution(labels):
    """
    Plots the distribution of samples in target variable (categorical)
    
    Input:
    - labels : list, list of target values
    
    """
######################################

    w = pd.value_counts(labels)

    # Barplot and font specifications
    barplot_specs = {"color": color_2, "alpha": 0.7, "edgecolor": "grey"}
    label_specs   = {"fontsize": 12}
    title_specs   = {"fontsize": 14, "fontweight": "bold", "y": 1.02}

    plt.figure(figsize=(8,4.5))
    sns.barplot(x=w.index,y=w.values, **barplot_specs);
    plt.ylabel('Counts',**label_specs);
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title('Sample distribution in target variable',**title_specs);

######################################    
def plot_loss_accuracy_curves(history):
######################################    

    title_specs = {"fontsize": 16}
    label_specs = {"fontsize": 14}

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Plot loss values
    ax1.set_title('Validation loss: {:.4f}'.format(history.history['val_loss'][-1]))
    ax1.plot(history.history['loss'], color =color_1, label='training set')
    ax1.plot(history.history['val_loss'], color =color_2, label='validation set')
    ax1.set_xlabel('Epochs',**label_specs)
    ax1.set_ylabel('Loss',**label_specs)
    ax1.set_ylim([0,None])
    ax1.legend()

    # plot accuracy values
    ax2.set_title('Validation accuracy: {:.2f}%'.format(history.history['val_accuracy'][-1]*100))
    ax2.plot(history.history['accuracy'], color =color_1, label='training set')
    ax2.plot(history.history['val_accuracy'],  color =color_2, label='validation set')
    ax2.set_xlabel('Epochs',**label_specs)
    ax2.set_ylabel('Accuracy',**label_specs)
    ax2.set_ylim([None,1])
    ax2.legend()

    plt.tight_layout()
    
######################################      
def plot_confusion_matrix(y, y_pred, labels, suptitle):
######################################    

    # Create two subplots
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    title_specs   = {"fontsize": 14, "fontweight": "bold", "y": 1.03}    
    plt.suptitle(suptitle,**title_specs)

    # Plots the standard confusion matrix
    ax1.set_title("Confusion Matrix (counts)", y= 1.02)

    ConfusionMatrixDisplay.from_predictions(y, 
                                            y_pred, 
                                            display_labels=labels, 
                                            cmap=plt.cm.Blues,
                                            values_format='d',
                                            ax=ax1)

    ax1.set_xticklabels(labels = labels, rotation=90)


    # Plots the normalized confusion matrix
    ax2.set_title("Confusion Matrix (ratios)", y= 1.02)

    ConfusionMatrixDisplay.from_predictions(y, 
                                            y_pred, 
                                            normalize="true", 
                                            display_labels=labels,
                                            cmap=plt.cm.Blues,
                                            values_format='.1g',
                                            ax=ax2)

    ax2.set_xticklabels(labels = labels, rotation=90)

    plt.tight_layout()