import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

from .generic_model import GenericModelPipeline, CellLineModel, EarlyStopping

from ..utils import *

class GATNet(torch.nn.Module, GenericModelPipeline):
    def __init__(self, num_features_xd=77, n_output=1,
                 xd_output_dim=128, xcl_output_dim=256,
                 dropout_gcn=0.2, dropout_att=0.1, dropout_mlp=0.1,
                 lr=1e-3, num_epochs=1000, weight_decay=0.01,
                 patience=5, delta=0, verbose=False, **kwargs):
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
        self.fc1 = nn.Linear(xcl_output_dim + 2 * xd_output_dim + len(TARGET_CLASSES), 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.out = nn.Linear(256 + len(TARGET_CLASSES), n_output) # Skip connection for cell target

        # Initialize the weights
        self._initialize_weights()
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_mlp)
        self.dropout_gcn = dropout_gcn
        
        # Optimizers
        self.verbose = verbose
        self.optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), lr=lr, weight_decay=weight_decay)
        self.num_epochs = num_epochs
        self.early_stopping = EarlyStopping(checkpoint_path=os.path.join(CHECKPOINT_DIR, 'best_gat'),
                                            patience=patience, delta=delta, verbose=self.verbose)
        self.criterion = nn.MSELoss()
        
    def _initialize_weights(self):
        # Initialize weights using He Initialization or Xavier, depending on the layer
        init.kaiming_normal_(self.gcn1_d1.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.gcn2_d1.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.gcn1_d2.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.gcn2_d2.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_normal_(self.out.weight)
        
    def _reset_batch_indices(self, batch):
        # Create a dictionary to map old batch indices to new batch indices
        index_map = {old_index.item(): new_index for new_index, old_index in enumerate(torch.sort(torch.unique(batch)).values)}

        # Reset batch using the mapping
        reset_batch = torch.tensor([index_map[old_index.item()] for old_index in batch], dtype=torch.long)

        # Return new batch indices along with the size for graph pooling
        return reset_batch, len(index_map)

    def forward(self, data):
        # Feed graph network for Drug 1
        xd1, edge_index1, batch_d1 = data.xd1, data.edge_index1, data.batch_d1
        batch_d1, batch_size = self._reset_batch_indices(batch_d1)
        xd1 = F.dropout(xd1, p=self.dropout_gcn, training=self.training)
        xd1 = F.elu(self.gcn1_d1(xd1, edge_index1))
        xd1 = F.dropout(xd1, p=self.dropout_gcn, training=self.training)
        xd1 = self.gcn2_d1(xd1, edge_index1)
        xd1 = self.relu(xd1)
        xd1 = gmp(xd1, batch_d1, size=batch_size)
        xd1 = self.fc_g1_d1(xd1)
        xd1 = self.relu(xd1)
        
        # Feed graph network for Drug 2
        xd2, edge_index2, batch_d2 = data.xd2, data.edge_index2, data.batch_d2
        batch_d2, batch_size = self._reset_batch_indices(batch_d2)
        xd2 = F.dropout(xd2, p=self.dropout_gcn, training=self.training)
        xd2 = F.elu(self.gcn1_d2(xd2, edge_index2))
        xd2 = F.dropout(xd2, p=self.dropout_gcn, training=self.training)
        xd2 = self.gcn2_d2(xd2, edge_index2)
        xd2 = self.relu(xd2)
        xd2 = gmp(xd2, batch_d2, size=batch_size)
        xd2 = self.fc_g1_d2(xd2)
        xd2 = self.relu(xd2)

        # Cell line features and cell line target; Reshape them first as it's collated
        xc1, xc2, xc3, xtc = data.xc1, data.xc2, data.xc3, data.xtc
        xc1 = xc1.view(-1, 1, 23808)
        xc2 = xc2.view(-1, 1, 3171)
        xc3 = xc3.view(-1, 1, 627)
        xtc = xtc.view(-1, 1, len(TARGET_CLASSES))
        xcl = self.cell_line_model(torch.cat((xc1, xc2, xc3, xtc), dim=2))
        
        # Concat processed cell line and drug features; proceed with FCN
        xtc = xtc.view(-1, len(TARGET_CLASSES))
        xc = torch.cat((xd1, xd2, xcl, xtc), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = torch.cat((xc, xtc), dim=1) # Employ skip connection for target cell
        out = self.out(xc)
        
        # Limit regression between -100 to 100
        out = torch.clamp(out, min=-100, max=100)

        return out
        
    def run_train(self, train_loader, val_loader, **kwargs):
        # TODO: Implement based on given
        self.to(DEVICE)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                inputs, targets = data.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Calculate average training loss for the epoch
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)

            # Validate the model
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    inputs, targets = data.to(DEVICE)
                    outputs = self(inputs)
                    val_loss += self.criterion(outputs, targets).item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            logging.debug(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Check for early stopping
            best_val_loss = min(val_loss, best_val_loss)
            self.early_stopping(val_loss, self)
            if self.early_stopping.early_stop:
                self.load_state_dict(torch.load(self.early_stopping.checkpoint))
                logging.info("Early stopping occurs")
                break

        logging.debug(f'Finished training with best validation loss: {best_val_loss}')

    def perform_prediction(self, X_test):
        self.eval()
        total_preds = torch.Tensor()
        with torch.no_grad():
            for data in X_test:
                data = data.to(DEVICE)
                output = self(data)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
        return total_preds.numpy().flatten()
