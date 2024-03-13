import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

from .generic_model import GenericModelPipeline, CellLineModel

# GAT model
class GATNet(torch.nn.Module, GenericModelPipeline):
    def __init__(self, num_features_xd=78, n_output=1,
                 xd_output_dim=128, xcl_output_dim=256,
                 n_filters=32, embed_dim=128,
                 dropout_gcn=0.2, dropout_att=0.1, dropout_mlp=0.1):
        super(GATNet, self).__init__()

        # Graph layers (both drugs use same architecture)
        self.gcn1_d1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout_att)
        self.gcn2_d1 = GATConv(num_features_xd * 10, xd_output_dim, dropout=dropout_att)
        self.fc_g1_d1 = nn.Linear(xd_output_dim, xd_output_dim)
        
        self.gcn1_d2 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout_att)
        self.gcn2_d2 = GATConv(num_features_xd * 10, xd_output_dim, dropout=dropout_att)
        self.fc_g1_d2 = nn.Linear(xd_output_dim, xd_output_dim)

        # Cell line features
        self.cell_line_model = CellLineModel(output_dim=xcl_output_dim, dropout=dropout_mlp)

        # MLP (Two graph outputs, cell line, and cell target)
        self.fc1 = nn.Linear(xcl_output_dim + 2 * xd_output_dim + 1, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.out = nn.Linear(256 + 1, n_output) # Skip connection for cell target

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_mlp)
        self.dropout_gcn = dropout_gcn

    def forward(self, data):
        # Gather input from data
        xd1, xd2, edge_index1, edge_index2, xcl, xct, batch = data.xd1, data.xd2, data.edge_index1, data.edge_index2, data.xcl, data.xct, data.batch
        
        # Feed graph network for Drug 1
        xd1 = F.dropout(xd1, p=self.dropout_gcn, training=self.training)
        xd1 = F.elu(self.gcn1_d1(xd1, edge_index1))
        xd1 = F.dropout(xd1, p=self.dropout_gcn, training=self.training)
        xd1 = self.gcn2_d1(xd1, edge_index1)
        xd1 = self.relu(xd1)
        xd1 = gmp(xd1, batch)
        xd1 = self.fc_g1_d1(xd1)
        xd1 = self.relu(xd1)
        
        # Feed graph network for Drug 2
        xd2 = F.dropout(xd2, p=self.dropout_gcn, training=self.training)
        xd2 = F.elu(self.gcn1_d2(xd2, edge_index2))
        xd2 = F.dropout(xd2, p=self.dropout_gcn, training=self.training)
        xd2 = self.gcn2_d2(xd2, edge_index2)
        xd2 = self.relu(xd2)
        xd2 = gmp(xd2, batch)
        xd2 = self.fc_g1_d2(xd2)
        xd2 = self.relu(xd2)

        # Cell line features and cell line target
        xcl = self.cell_line_model(torch.cat((xcl, xct), dim=1))
        
        # Concat processed cell line and drug features; proceed with FCN
        xc = torch.cat((xd1, xd2, xcl, xct), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = torch.cat((xc, xct), dim=1) # Employ skip connection for target cell
        out = self.out(xc)
        
        # Limit regression between -100 to 100
        out = nn.Sigmoid()(out)
        out = (sigmoid_out * 200) - 100

        return out
        
    def run_train(self, X_train, y_train):
        # TODO: Implement based on given
        pass


    def perform_prediction(self, X_test):
        self.eval()
        total_preds = torch.Tensor()
        with torch.no_grad():
            for data in X_test:
                data = data.to(DEVICE)
                output = self(data)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
        return total_preds.numpy().flatten()