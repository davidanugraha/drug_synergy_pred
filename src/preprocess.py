# Preprocess DataLoaders
from tdc.benchmark_group import drugcombo_group

import torch
import numpy as np
import pandas as pd
from rdkit import Chem

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch_geometric import data as DATA
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
from .utils import *

ELEMENT_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 
                'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Other']
DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUMBER_OF_H_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
IMPLICIT_VALENCE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

BENCHMARK_GROUP = drugcombo_group(path=DATASET_DIR)
BENCHMARK = BENCHMARK_GROUP.get('Drugcomb_CSS')
BENCHMARK_NAME = BENCHMARK['name']


class BatchCustomGraphData:
    def __init__(self, xd1, edge_index1, xsmile1, xd2, edge_index2, xsmile2, batch_d1, batch_d2, xc1, xc2, xc3, xtc, labels, smile_graph, saliency_map=False):
        self.xd1 = xd1
        self.edge_index1 = edge_index1
        self.xsmile1 = xsmile1
        self.xd2 = xd2
        self.edge_index2 = edge_index2
        self.xsmile2 = xsmile2
        self.batch_d1 = batch_d1
        self.batch_d2 = batch_d2
        self.xc1 = xc1
        self.xc2 = xc2
        self.xc3 = xc3
        self.xtc = xtc
        self.labels = labels
        self.smile_graph = smile_graph
        self.saliency_map = saliency_map
        
    def to(self, device):
        self.xd1 = self.xd1.to(device)
        self.edge_index1 = self.edge_index1.to(device)
        self.xsmile1 = self.xsmile1.to(device)
        self.xd2 = self.xd2.to(device)
        self.edge_index2 = self.edge_index2.to(device)
        self.xsmile2 = self.xsmile2.to(device)
        self.batch_d1 = self.batch_d1.to(device)
        self.batch_d2 = self.batch_d2.to(device)
        self.xc1 = self.xc1.to(device)
        self.xc2 = self.xc2.to(device)
        self.xc3 = self.xc3.to(device)
        self.xtc = self.xtc.to(device)
        self.labels = self.labels.to(device)

        return self

class DrugCSSDataset(Dataset):
    def __init__(self, xd1, xd2, xc1, xc2, xc3, xtc, y, smile_graph, smile_feature, saliency_map=False):
        self.xd1 = xd1
        self.xd2 = xd2
        self.xc1 = xc1
        self.xc2 = xc2
        self.xc3 = xc3
        self.xtc = xtc
        self.y = y
        self.smile_graph = smile_graph
        self.smile_feature = smile_feature
        self.saliency_map = saliency_map

    def __len__(self):
        return len(self.xd1)

    def __getitem__(self, idx):
        features1, edge_index1 = self.smile_graph[self.xd1[idx]]
        smile_feature1 = self.smile_feature[self.xd1[idx]]
        features2, edge_index2 = self.smile_graph[self.xd2[idx]]
        smile_feature2 = self.smile_feature[self.xd2[idx]]

        # Create Data object for both Drug 1 and Drug 2
        data = DATA.Data(
            xd1=torch.tensor(features1, dtype=torch.float),
            edge_index1=torch.tensor(edge_index1, dtype=torch.long).t().contiguous(),
            xsmile1 = torch.tensor(smile_feature1, dtype=torch.float),
            xd2=torch.tensor(features2, dtype=torch.float),
            edge_index2=torch.tensor(edge_index2, dtype=torch.long).t().contiguous(),
            xsmile2 = torch.tensor(smile_feature2, dtype=torch.float),
            xc1=torch.tensor(self.xc1[idx], dtype=torch.float),
            xc2=torch.tensor(self.xc2[idx], dtype=torch.float),
            xc3=torch.tensor(self.xc3[idx], dtype=torch.float),
            xtc=torch.tensor(self.xtc[idx], dtype=torch.float),
            labels=torch.tensor(self.y[idx], dtype=torch.float),
            smile_graph=self.smile_graph,
            saliency_map=self.saliency_map
        )

        return data

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
        # feature = (one_hot_encoding(atom.GetSymbol(), ELEMENT_LIST) + one_hot_encoding(atom.GetDegree(), DEGREE_LIST) +
        #            one_hot_encoding(atom.GetTotalNumHs(), NUMBER_OF_H_LIST) + one_hot_encoding(atom.GetImplicitValence(), IMPLICIT_VALENCE_LIST))
        feature = [atom.GetAtomicNum(), atom.GetMass(),atom.GetDegree(),atom.GetTotalNumHs(), atom.GetImplicitValence(),atom.GetExplicitValence(),
               atom.GetFormalCharge(),atom.GetIsAromatic(), atom.GetIsotope(), atom.GetNoImplicit(), atom.GetNumExplicitHs(),atom.GetNumImplicitHs(),
               atom.GetNumRadicalElectrons(),atom.GetTotalDegree(), atom.GetTotalValence(), atom.InvertChirality(), atom.IsInRing()
               ]
        # Convert features to integers and append features
        # feature = [int(v) for v in feature]
        feature = [int(round(v)) for v in feature]
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

rdkit_features =  ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'PEOE_VSA14', 'SMR_VSA10', 'SlogP_VSA12', 'EState_VSA11', 'VSA_EState10', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'BalabanJ', 'BertzCT', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9']
def _calculate_property_rdkit(smiles, property_name):
    molecule = Chem.MolFromSmiles(smiles)
    try:
        property_value = getattr(Descriptors, property_name)(molecule)
        return property_value
    except AttributeError:
        print(f"Error: Descriptor '{property_name}' does not exist.")
        return None

def calculate_rdkit_features(smiles, properties):
    return [_calculate_property_rdkit(smiles, prop) for prop in properties]

def calculate_mordred_features(smile_string):
    # Get mordred features
    calc = Calculator(descriptors, ignore_3D=True)
    
    # Convert smile to molecule
    molecule = Chem.MolFromSmiles(smile_string)
    if not molecule:
        raise ValueError("Invalid SMILES string provided.")

    descriptor_values = calc(molecule)
    
    # fill 0 to None
    values_list = []
    for value in descriptor_values.values():
        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            values_list.append(value)
        else:
            values_list.append(0) 
    return values_list

def get_smile_features(smile):
    return calculate_rdkit_features(smile, rdkit_features) + calculate_mordred_features(smile)

# Add graph to the dataframe
def add_graph(data, smile_to_graph_dict, smile_to_smile_feature_dict):
    features_list1 = []
    edges_list1 = []
    smile_feature_list1 = []
    for drug_smiles in data['Drug1']:
        features, edges = smile_to_graph_dict.get(drug_smiles, (None, None))
        features_list1.append(features)
        edges_list1.append(edges)
        smile_feature_list1.append(smile_to_smile_feature_dict.get(drug_smiles, (None, None)))
        
    features_list2 = []
    edges_list2 = []
    smile_feature_list2 = []
    for drug_smiles in data['Drug2']:
        features, edges = smile_to_graph_dict.get(drug_smiles, (None, None))
        features_list2.append(features)
        edges_list2.append(edges)
        smile_feature_list2.append(smile_to_smile_feature_dict.get(drug_smiles, (None, None)))
        
    data['Drug1_Atom_Feature'], data['Drug1_Atom_Edges'] = features_list1, edges_list1
    data['Drug2_Atom_Feature'], data['Drug2_Atom_Edges'] = features_list2, edges_list2
    data['Drug1_Mol_Feature'], data['Drug2_Mol_Feature'] = smile_feature_list1, smile_feature_list2

    
def prepare_dataframe(val_split):
    # Gather data from TDC Benchmark
    train_val_df, test_df = BENCHMARK['train_val'], BENCHMARK['test']
    
    # Unpack CellLine arrays into 3 distinct arrays
    train_val_df[['CellLine1', 'CellLine2', 'CellLine3']] = train_val_df['CellLine'].apply(lambda x: pd.Series([x[0], x[1], x[2]]))
    test_df[['CellLine1', 'CellLine2', 'CellLine3']] = test_df['CellLine'].apply(lambda x: pd.Series([x[0], x[1], x[2]]))
    
    # Create SMILES string to graph dictionary and Smile Features
    smile_to_graph_dict = {}
    smile_to_smile_feature_dict = {}
    all_drugs = np.unique(np.concatenate((train_val_df['Drug1'].unique(),
                                          train_val_df['Drug2'].unique(),
                                          test_df['Drug1'].unique(),
                                          test_df['Drug2'].unique())))
    for drug in all_drugs:
        smile_to_graph_dict[drug] = smile_to_graph(drug)
        smile_to_smile_feature_dict[drug] = get_smile_features(drug)
    
    # Add graph train, val, and test
    add_graph(train_val_df, smile_to_graph_dict, smile_to_smile_feature_dict)
    add_graph(test_df, smile_to_graph_dict, smile_to_smile_feature_dict)
    
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
    return train_df, val_df, test_df, smile_to_graph_dict, smile_to_smile_feature_dict

def custom_collate(batch):
    # Stack the input tensors along the batch dimension
    xd1 = torch.vstack([data.xd1 for data in batch])
    edge_index1 = torch.hstack([data.edge_index1 for data in batch])
    xsmile1 = torch.hstack([data.xsmile1 for data in batch])
    xd2 = torch.vstack([data.xd2 for data in batch])
    edge_index2 = torch.hstack([data.edge_index2 for data in batch])
    xsmile2 = torch.hstack([data.xsmile2 for data in batch])
    xc1 = torch.vstack([data.xc1 for data in batch])
    xc2 = torch.vstack([data.xc2 for data in batch])
    xc3 = torch.vstack([data.xc3 for data in batch])
    xtc = torch.vstack([data.xtc for data in batch])
    labels = torch.vstack([data.labels for data in batch])
    smile_graph = [data.smile_graph for data in batch]
    saliency_map = [data.saliency_map for data in batch]
    
    # Compute batch_d1 and batch_d2
    batch_d1 = []
    batch_d2 = []
    for i, data in enumerate(batch):
        batch_d1.extend([i] * data.xd1.size(0))
        batch_d2.extend([i] * data.xd2.size(0))
    batch_d1 = torch.tensor(batch_d1, dtype=torch.long)
    batch_d2 = torch.tensor(batch_d2, dtype=torch.long)

    return BatchCustomGraphData(xd1=xd1, edge_index1=edge_index1,xsmile1=xsmile1, xd2=xd2, edge_index2=edge_index2, xsmile2=xsmile2,
                                batch_d1=batch_d1, batch_d2=batch_d2, xc1=xc1, xc2=xc2, xc3=xc3,
                                xtc=xtc, labels=labels, smile_graph=smile_graph, saliency_map=saliency_map)

def get_data_loaders(val_split=0.2, batch_size=32):
    # Prepare dataframe
    train_df, val_df, test_df, smile_to_graph_dict, smile_to_smile_feature_dict = prepare_dataframe(val_split)

    xd1, xd2, xc1, xc2, xc3, xtc, y = extract_attributes_from_df(train_df)
    train_dataset = DrugCSSDataset(xd1=xd1, xd2=xd2, xc1=xc1, xc2=xc2, xc3=xc3, xtc=xtc, y=y, smile_graph=smile_to_graph_dict, smile_feature = smile_to_smile_feature_dict)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    xd1, xd2, xc1, xc2, xc3, xtc, y = extract_attributes_from_df(val_df)
    val_dataset = DrugCSSDataset(xd1=xd1, xd2=xd2, xc1=xc1, xc2=xc2, xc3=xc3, xtc=xtc, y=y, smile_graph=smile_to_graph_dict, smile_feature = smile_to_smile_feature_dict)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    xd1, xd2, xc1, xc2, xc3, xtc, y = extract_attributes_from_df(test_df)
    test_dataset = DrugCSSDataset(xd1=xd1, xd2=xd2, xc1=xc1, xc2=xc2, xc3=xc3, xtc=xtc, y=y, smile_graph=smile_to_graph_dict, smile_feature = smile_to_smile_feature_dict)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return train_loader, val_loader, test_loader
# def evaluate_benchmark(predictions):
#     out = BENCHMARK_GROUP.evaluate(predictions)
#     logging.info(out)