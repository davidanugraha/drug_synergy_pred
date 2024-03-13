from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class GenericModelPipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run_train(self, X_train, y_train):
        """
        Train the model using the preprocessed data with whatever training algorithm used.

        Parameters:
        - X_train: Input features
        - y_train: Target labels
        """
        pass

    @abstractmethod
    def perform_prediction(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        - X_test: Test input features

        Returns:
        - Predicted values (y_test)
        """
        pass

class CellLineModel(torch.nn.Module):
    def __init__(self, n_filters_gene=3, n_filters_proteomic=5, n_filters_rna=7, output_dim=256, dropout=0.1):
        super(CellLineModel, self).__init__()

        # Define Convolution Layer for gene features (input_dim = 23808)
        self.conv_xg_1 = nn.Conv1d(in_channels=1, out_channels=n_filters_gene, kernel_size=11, stride=2)
        self.bn_xg_1 = nn.BatchNorm1d(n_filters_gene)
        self.pool_xg_1 = nn.MaxPool1d(5)
        self.conv_xg_2 = nn.Conv1d(in_channels=n_filters_gene, out_channels=n_filters_gene * 2, kernel_size=5, stride=2)
        self.bn_xg_2 = nn.BatchNorm1d(n_filters_gene * 2)
        self.pool_xg_2 = nn.MaxPool1d(5, stride=2)
        self.conv_xg_3 = nn.Conv1d(in_channels=n_filters_gene * 2, out_channels=n_filters_gene, kernel_size=5, stride=2)
        self.bn_xg_3 = nn.BatchNorm1d(n_filters_gene)
        self.pool_xg_3 = nn.MaxPool1d(5, stride=2)
    
        # Define convolution layer for proteomic features (input_dim = 3171)
        self.conv_xp_1 = nn.Conv1d(in_channels=1, out_channels=n_filters_proteomic, kernel_size=5, stride=2)
        self.bn_xp_1 = nn.BatchNorm1d(n_filters_proteomic)
        self.pool_xp_1 = nn.MaxPool1d(5)
        self.conv_xp_2 = nn.Conv1d(in_channels=n_filters_gene, out_channels=n_filters_proteomic * 2, kernel_size=5, stride=2)
        self.bn_xp_2 = nn.BatchNorm1d(n_filters_proteomic * 2)
        self.pool_xp_2 = nn.MaxPool1d(5, stride=2)
        self.conv_xp_3 = nn.Conv1d(in_channels=n_filters_gene * 2, out_channels=n_filters_proteomic * 3, kernel_size=5, stride=2)
        self.bn_xp_3 = nn.BatchNorm1d(n_filters_proteomic * 3)
        self.pool_xp_3 = nn.MaxPool1d(5, stride=2)

        # Define convolution layer for microRNA features (input_dim = 627)
        self.conv_xm_1 = nn.Conv1d(in_channels=1, out_channels=n_filters_rna, kernel_size=5)
        self.bn_xm_1 = nn.BatchNorm1d(n_filters_rna)
        self.pool_xm_1 = nn.MaxPool1d(5)
        self.conv_xm_2 = nn.Conv1d(in_channels=n_filters_gene, out_channels=n_filters_rna * 2, kernel_size=5, stride=2)
        self.bn_xm_2 = nn.BatchNorm1d(n_filters_rna * 2)
        self.pool_xm_2 = nn.MaxPool1d(5, stride=2)
        self.conv_xm_3 = nn.Conv1d(in_channels=n_filters_gene * 2, out_channels=n_filters_rna * 3, kernel_size=5, stride=2)
        self.bn_xm_3 = nn.BatchNorm1d(n_filters_rna * 3)
        self.pool_xm_3 = nn.MaxPool1d(5, stride=2)
        
        # Concatenated layer for MLP along with target cell; +1 is to introduce target cell again
        self.fc1 = nn.Linear(2220 + 1440 + 756 + 1, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2 + 1, output_dim)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Separate x into size 23808, 3171, 627, 1
        xg, xp, xm, xtc = x[:, :23808], x[:, 23808:26979], x[:, 26979:27606], x[:, -1].unsqueeze(1)
        
        # Apply for gene features
        xg = self.conv_xg_1(xg)
        xg = self.bn_xg_1(xg)
        xg = self.relu(xg)
        xg = self.pool_xg_1(xg)

        xg = self.conv_xg_2(xg)
        xg = self.bn_xg_2(xg)
        xg = self.relu(xg)
        xg = self.pool_xg_2(xg)

        xg = self.conv_xg_3(xg)
        xg = self.bn_xg_3(xg)
        xg = self.relu(xg)
        xg = self.pool_xg_3(xg)

        xg = xg.view(xg.size(0), -1)  # Flatten
        
        # Apply for proteomic features
        xp = self.conv_xp_1(xp)
        xp = self.bn_xp_1(xp)
        xp = self.relu(xp)
        xp = self.pool_xp_1(xp)

        xp = self.conv_xp_2(xp)
        xp = self.bn_xp_2(xp)
        xp = self.relu(xp)
        xp = self.pool_xp_2(xp)

        xp = self.conv_xp_3(xp)
        xp = self.bn_xp_3(xp)
        xp = self.relu(xp)
        xp = self.pool_xp_3(xp)

        xp = xp.view(xp.size(0), -1)  # Flatten
        
        # Apply for microRNA features
        xm = self.conv_xm_1(xm)
        xm = self.bn_xm_1(xm)
        xm = self.relu(xm)
        xm = self.pool_xm_1(xm)

        xm = self.conv_xm_2(xm)
        xm = self.bn_xm_2(xm)
        xm = self.relu(xm)
        xm = self.pool_xm_2(xm)

        xm = self.conv_xm_3(xm)
        xm = self.bn_xm_3(xm)
        xm = self.relu(xm)
        xm = self.pool_xm_3(xm)

        xm = xm.view(xm.size(0), -1)  # Flatten
        
        # Concatenate the newly trained features with target cell information
        concat_features = torch.cat((xg, xp, xm, xtc), dim=1)
        
        # Fully connected layers
        fc1_out = self.fc1(concat_features)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        fc1_out = torch.cat((fc1_out, xtc), dim=1)
        
        output = self.fc2(fc1_out)
        
        return output
        
        


