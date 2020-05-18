import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split,DataLoader, RandomSampler, SequentialSampler
import torch.utils.data as data_utils
import torch.optim as optim
import gc #garbage collector for gpu memory 
from tqdm import tqdm
from collections import Counter
from transformers import BertForSequenceClassification, BertTokenizer


train_path = '../liar_dataset/train.tsv'
test_path = '../liar_dataset/test.tsv'
val_path = '../liar_dataset/valid.tsv'

bert_length = 512

def to_onehot(a):
    a_cat = [0]*len(a)
    for i in range(len(a)):
        if a[i]=='true':
            a_cat[i] = 0
        elif a[i]=='mostly-true':
            a_cat[i] = 1
        elif a[i]=='half-true':
            a_cat[i] = 2
        elif a[i]=='barely-true':
            a_cat[i] = 3
        elif a[i]=='false':
            a_cat[i] = 4
        elif a[i]=='pants-fire':
            a_cat[i] = 5
        else:
            print('Incorrect label')
    return a_cat

def build_dataset(statements,labels,length):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    input_ids = []
    attention_masks = []

    for sentance in statements:
        encoded_dict = tokenizer.encode_plus(sentance,add_special_tokens = True,max_length = length,
        pad_to_max_length = True,return_attention_mask = True,return_tensors = 'pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    # print(input_ids.shape)
    # print(attention_masks.shape)
    # print(labels.shape)

    dataset = TensorDataset(input_ids,attention_masks,labels)
    return dataset


def get_liar_dataset():
    train_df = pd.read_csv(train_path, sep="\t", header=None)
    test_df = pd.read_csv(test_path, sep="\t", header=None)
    val_df = pd.read_csv(val_path, sep="\t", header=None)

    train = train_df.values
    test = test_df.values
    val = val_df.values

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #print(Counter(train[:,1]))

    labels = {'train':train[:,1], 'test':test[:,1], 'val':val[:,1]}
    statements = {'train':train[:,2], 'test':test[:,2], 'val':val[:,2]}
    subjects = {'train':train[:,3], 'test':test[:,3], 'val':val[:,3]}
    speaker = {'train':train[:,4], 'test':test[:,4], 'val':val[:,4]}
    job = {'train':train[:,5], 'test':test[:,5], 'val':val[:,5]}
    state = {'train':train[:,6], 'test':test[:,6], 'val':val[:,6]}
    affiliation = {'train':train[:,7], 'test':test[:,7], 'val':val[:,7]}
    # credit = {'train':[train[i][9:14] for i in range(len(train))], 'test':[test[i][9:14] for i in range(len(test))], 'val':[val[i][9:14] for i in range(len(val))]}
    # context = {'train':[train[i][14] for i in range(len(train))], 'test':[test[i][14] for i in range(len(test))], 'val':[val[i][14] for i in range(len(val))]}
    # justification = {'train':[train[i][15] for i in range(len(train))], 'test':[test[i][15] for i in range(len(test))], 'val':[val[i][15] for i in range(len(val))]}

    labels_onehot = {'train':to_onehot(labels['train']), 'test':to_onehot(labels['test']), 'val':to_onehot(labels['val'])}

    train_dataset = build_dataset(statements['train'],labels_onehot['train'],bert_length)
    val_dataset = build_dataset(statements['val'],labels_onehot['val'],bert_length)
    test_dataset = build_dataset(statements['test'],labels_onehot['test'],bert_length)

    return train_dataset, val_dataset, test_dataset
