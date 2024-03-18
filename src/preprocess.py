# Preprocess DataLoaders
import tdc

import torch
import numpy as np
import pandas as pd
from rdkit import Chem

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA

from .utils import *

ELEMENT_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 
                'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Other']
DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUMBER_OF_H_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
IMPLICIT_VALENCE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

BENCHMARK_GROUP = tdc.BenchmarkGroup('drugcombo_group', path=DATASET_DIR, file_format='pkl')
BENCHMARK_NAME = 'drugcomb_css'

class CustomGraphData(DATA.Data): 
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index1':
            return self.xd1.size(0)
        if key == 'edge_index2':
            return self.xd2.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class DrugCSSDataset(InMemoryDataset):
    def __init__(self, xd1, xd2, xc1, xc2, xc3, xtc, y, smile_graph, saliency_map=False):
        self.xd1 = xd1
        self.xd2 = xd2
        self.xc1 = xc1
        self.xc2 = xc2
        self.xc3 = xc3
        self.xtc = xtc
        self.y = y
        self.smile_graph = smile_graph
        self.saliency_map = saliency_map
        super(DrugCSSDataset, self).__init__(root=DATASET_DIR, transform=None, pre_transform=None, pre_filter=None)
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
            # print('Converting SMILES to graph: {}/{}'.format(i + 1, len(self.xd1)))
            features1, edge_index1 = self.smile_graph[self.xd1[i]]
            features2, edge_index2 = self.smile_graph[self.xd2[i]]
            
            # Create Data object for both Drug 1 and Drug 2
            data = CustomGraphData(
                xd1=torch.tensor(features1, dtype=torch.float),
                edge_index1=torch.tensor(edge_index1, dtype=torch.long).t().contiguous(),
                xd2=torch.tensor(features2, dtype=torch.float),
                edge_index2=torch.tensor(edge_index2, dtype=torch.long).t().contiguous(),
                xc1=torch.tensor(self.xc1[i], dtype=torch.float),
                xc2=torch.tensor(self.xc2[i], dtype=torch.float),
                xc3=torch.tensor(self.xc3[i], dtype=torch.float),
                xtc=torch.tensor(self.xtc[i], dtype=torch.float),
                batch_d1=torch.tensor([i] * len(features1), dtype=torch.long),
                batch_d2=torch.tensor([i] * len(features2), dtype=torch.long),
                labels=torch.tensor(self.y[i], dtype=torch.float),
                smile_graph=self.smile_graph,
                saliency_map=self.saliency_map
            )
            
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def one_hot_encoding(x, l):
    # If x is not in the list, use the last element
    if x not in l:
        x = l[-1]
    # Return one-hot encoded list
    return list(map(lambda s: x == s, l))

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    atom_features = []
    for atom in molecule.GetAtoms():
        # Create features for each atom based on its symbol, degree, number of Hs, and implicit valence
        feature = (one_hot_encoding(atom.GetSymbol(), ELEMENT_LIST) + one_hot_encoding(atom.GetDegree(), DEGREE_LIST) +
                   one_hot_encoding(atom.GetTotalNumHs(), NUMBER_OF_H_LIST) + one_hot_encoding(atom.GetImplicitValence(), IMPLICIT_VALENCE_LIST))

        # Convert features to integers and append features
        feature = [int(v) for v in feature]
        atom_features.append(feature)
    
    # Add edges between atoms
    edges = []
    for bond in molecule.GetBonds():
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        edges.append([start_atom_idx, end_atom_idx])

    return atom_features, edges

# Extract each attributes into a list
def extract_attributes_from_df(df):
    data = df[['Drug1', 'Drug2', 'CellLine1', 'CellLine2', 'CellLine3', 'Y']].to_numpy()
    xd1 = data[:, 0]
    xd2 = data[:, 1]
    xc1 = data[:, 2]
    xc2 = data[:, 3]
    xc3 = data[:, 4]
    y = data[:, 5]
    xtc = df[TARGET_CLASSES].to_numpy()
    return xd1, xd2, xc1, xc2, xc3, xtc, y

# Add graph to the dataframe
def add_graph(data, smile_to_graph_dict):
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

def prepare_dataframe(val_split):
    # Gather data from TDC Benchmark
    train_val_df, test_df = BENCHMARK_GROUP.get(BENCHMARK_NAME)['train_val'], BENCHMARK_GROUP.get(BENCHMARK_NAME)['test']
    
    # Unpack CellLine arrays into 3 distinct arrays
    train_val_df[['CellLine1', 'CellLine2', 'CellLine3']] = train_val_df['CellLine'].apply(lambda x: pd.Series([x[0], x[1], x[2]]))
    test_df[['CellLine1', 'CellLine2', 'CellLine3']] = test_df['CellLine'].apply(lambda x: pd.Series([x[0], x[1], x[2]]))
    
    # Create SMILES string to graph dictionary
    smile_to_graph_dict = {}
    all_drugs = np.unique(np.concatenate((train_val_df['Drug1'].unique(),
                                          train_val_df['Drug2'].unique(),
                                          test_df['Drug1'].unique(),
                                          test_df['Drug2'].unique())))
    for drug in all_drugs:
        smile_to_graph_dict[drug] = smile_to_graph(drug)
    
    # Add graph train, val, and test
    add_graph(train_val_df, smile_to_graph_dict)
    add_graph(test_df, smile_to_graph_dict)
    
    # Perform train-validation split
    train_df, val_df = train_test_split(train_val_df, test_size=val_split, stratify=train_val_df['target_class'], random_state=RANDOM_SEED)

    # Replace target class with dummy variables
    train_df = pd.get_dummies(train_df, columns=['target_class'])
    val_df = pd.get_dummies(val_df, columns=['target_class'])
    test_df = pd.get_dummies(test_df, columns=['target_class'])
    
    # Ensure order consistency in dummy variable columns for all sets
    for class_name in TARGET_CLASSES:
        if class_name not in train_df.columns:
            train_df[class_name] = 0
        if class_name not in val_df.columns:
            val_df[class_name] = 0
        if class_name not in test_df.columns:
            test_df[class_name] = 0
    
    # Drop index and return
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df, smile_to_graph_dict
    
def get_data_loaders(val_split=0.2, batch_size=32):
    # Prepare dataframe
    train_df, val_df, test_df, smile_to_graph_dict = prepare_dataframe(val_split)

    xd1, xd2, xc1, xc2, xc3, xtc, y = extract_attributes_from_df(train_df)
    train_dataset = DrugCSSDataset(xd1=xd1, xd2=xd2, xc1=xc1, xc2=xc2, xc3=xc3, xtc=xtc, y=y, smile_graph=smile_to_graph_dict)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    xd1, xd2, xc1, xc2, xc3, xtc, y = extract_attributes_from_df(val_df)
    val_dataset = DrugCSSDataset(xd1=xd1, xd2=xd2, xc1=xc1, xc2=xc2, xc3=xc3, xtc=xtc, y=y, smile_graph=smile_to_graph_dict)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    xd1, xd2, xc1, xc2, xc3, xtc, y = extract_attributes_from_df(test_df)
    test_dataset = DrugCSSDataset(xd1=xd1, xd2=xd2, xc1=xc1, xc2=xc2, xc3=xc3, xtc=xtc, y=y, smile_graph=smile_to_graph_dict)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
