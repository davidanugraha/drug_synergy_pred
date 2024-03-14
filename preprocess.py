# Preprocess DataLoaders
import copy
import tdc
from model_classes import *

import torch
import numpy as np
import pandas as pd
import time
import pickle
import argparse
import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA

def one_hot_encoding(x, l):
    if x not in l:
        x = l[-1]
    return list(map(lambda s: x == s, l))
element_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 
                'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Other']
degree_list = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]
Number_of_H_list = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]
ImplicitValence_list = [0, 1, 2, 3, 4, 5, 6,7,8,9,10]

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    atom_features = []
    for atom in molecule.GetAtoms():
        feature = (one_hot_encoding(atom.GetSymbol(), element_list) + one_hot_encoding(atom.GetDegree(), degree_list) +
                   one_hot_encoding(atom.GetTotalNumHs(), Number_of_H_list) + one_hot_encoding(atom.GetImplicitValence(), ImplicitValence_list))
        feature = [int(v) for v in feature]
        atom_features.append(feature)
    
    edges = []
    for bond in molecule.GetBonds():
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        edges.append([start_atom_idx, end_atom_idx])

    return atom_features, edges

def extract(cellline, index):
    out = cellline[index]
    return out


    
def get_xd_xc(dataframe):
    xd1 = dataframe['Drug1'].tolist()
    xd2 = dataframe['Drug2'].tolist()
    xc1 = dataframe['Cell1'].tolist()
    xc2 = dataframe['Cell2'].tolist()
    xc3 = dataframe['Cell3'].tolist()
    y = dataframe['Y'].tolist()
    return xd1, xd2, xc1, xc2, xc3, y

def combine_two_drugs(feature1, edge1, feature2, edge2):
    combined_features = feature1 + feature2
    combined_edges = edge1 + (np.array(edge2)+len(feature1)).tolist()
    return combined_features, combined_edges

def prepare_data():
    group = tdc.BenchmarkGroup('drugcombo_group', path='data/',
                               file_format='pkl')
    name = 'drugcomb_css'
    train_val = group.get(name)['train_val']
    test = group.get(name)['test']
    train_val.head()
    All_Drugs = np.unique(np.concatenate((train_val['Drug1'].unique(), train_val['Drug2'].unique(), test['Drug1'].unique(), test['Drug2'].unique())))
    
    train_val['Cell1'] = train_val['CellLine'].apply(extract, args=(0,))
    train_val['Cell2'] = train_val['CellLine'].apply(extract, args=(1,))
    train_val['Cell3'] = train_val['CellLine'].apply(extract, args=(2,))
    test['Cell1'] = test['CellLine'].apply(extract, args=(0,))
    test['Cell2'] = test['CellLine'].apply(extract, args=(1,))
    test['Cell3'] = test['CellLine'].apply(extract, args=(2,))
    
    def Add_Graph(data):
        features_list1 = []
        edges_list1 = []
        for drug_smiles in data['Drug1']:
            features, edges = smile_to_graph_dict.get(drug_smiles, (None, None))
            features_list1.append(features)
            edges_list1.append(edges)
            
        features_list2 = []
        edges_list2 = []
        for drug_smiles in data['Drug2']:
            features, edges = smile_to_graph_dict.get(drug_smiles, (None, None))
            features_list2.append(features)
            edges_list2.append(edges)
            
        data['Drug1_Atom_Feature'], data['Drug1_Atom_Edges'] = features_list1, edges_list1
        data['Drug2_Atom_Feature'], data['Drug2_Atom_Edges'] = features_list2, edges_list2
    
    smile_to_graph_dict = {}
    for drug in All_Drugs:
        smile_to_graph_dict[drug] = smile_to_graph(drug)
    Add_Graph(train_val)
    Add_Graph(test)
    train_df, val_df = train_test_split(train_val, test_size=0.3, random_state=42)
    test_df = test
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df, smile_to_graph_dict
    
class TestDataset(InMemoryDataset):
    def __init__(self, root='/tmp', xd1=None, xd2=None, xc1=None, xc2 = None, xc3 = None, y=None, smile_graph=None, saliency_map=False):
        self.xd1 = xd1
        self.xd2 = xd2
        self.xc1 = xc1
        self.xc2 = xc2
        self.xc3 = xc3
        self.y = y
        self.smile_graph = smile_graph
        self.saliency_map = saliency_map
        # super(TestDataset, self).__init__(root)
        super(TestDataset, self).__init__(root=root, transform=None, pre_transform=None, pre_filter=None)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in range(len(self.xd1)):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, len(self.xd1)))
            
            d1, d2, c1, c2, c3, label = self.xd1[i], self.xd2[i], self.xc1[i], self.xc2[i],self.xc3[i], self.y[i]
            xc1_tensor, xc2_tensor, xc3_tensor = torch.tensor(c1, dtype=torch.float), torch.tensor(c2, dtype=torch.float), torch.tensor(c3, dtype=torch.float)
            features1, edge_index1 = self.smile_graph[d1]
            features2, edge_index2 = self.smile_graph[d2]
            combined_features, combined_edges = combine_two_drugs(features1, edge_index1, features2, edge_index2)
            data = DATA.Data(x=torch.tensor(combined_features, dtype=torch.float),
                             edge_index=torch.tensor(combined_edges, dtype=torch.long).t().contiguous(),
                             y=torch.tensor([label], dtype=torch.float),
                             cell1 = xc1_tensor,cell2 = xc2_tensor,cell3 = xc3_tensor)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
def get_DataLoaders():
    train_df, val_df, test_df, smile_to_graph_dict = prepare_data()
    root = '/Users/derrick/Desktop/'
    xd1, xd2, xc1, xc2, xc3, y = get_xd_xc(train_df)
    train_dataset = TestDataset(root=root, xd1=xd1, xd2=xd2, xc1=xc1,xc2=xc2,xc3=xc3, y=y, smile_graph=smile_to_graph_dict)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    xd1, xd2, xc1, xc2, xc3, y = get_xd_xc(val_df)
    val_dataset = TestDataset(root=root, xd1=xd1, xd2=xd2, xc1=xc1,xc2=xc2,xc3=xc3, y=y, smile_graph=smile_to_graph_dict)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    xd1, xd2, xc1, xc2, xc3, y = get_xd_xc(test_df)
    test_dataset = TestDataset(root=root, xd1=xd1, xd2=xd2, xc1=xc1,xc2=xc2,xc3=xc3, y=y, smile_graph=smile_to_graph_dict)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
