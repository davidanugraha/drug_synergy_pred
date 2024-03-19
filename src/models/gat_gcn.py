import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import mean_absolute_error, mean_squared_error

from .generic_model import GenericModelPipeline, CellLineModel, EarlyStopping

from ..utils import *

class GATGCNNet(torch.nn.Module, GenericModelPipeline):
    def __init__(self, num_features_xd=77, n_output=1, num_heads=10, gcn_linear_features=1500,
                 xd_output_dim=128, xcl_output_dim=256,
                 dropout_gcn=0.2, dropout_att=0.1, dropout_mlp=0.1,
                 lr=1e-3, num_epochs=1000, weight_decay=0.01,
                 patience=5, delta=0, verbose=False, **kwargs):
        super(GATGCNNet, self).__init__()

        # Graph layers (both drugs use same architecture)
        self.gat1_d1 = GATConv(num_features_xd, num_features_xd, heads=num_heads, dropout=dropout_att)
        self.gcn1_d1 = GCNConv(num_features_xd * num_heads, num_features_xd * num_heads)
        self.fc_g1_d1 = nn.Linear(num_features_xd * num_heads * 2, gcn_linear_features)
        self.fc_g2_d1 = nn.Linear(gcn_linear_features, xd_output_dim)
        
        self.gat1_d2 = GATConv(num_features_xd, num_features_xd, heads=num_heads, dropout=dropout_att)
        self.gcn1_d2 = GCNConv(num_features_xd * num_heads, num_features_xd * num_heads)
        self.fc_g1_d2 = nn.Linear(num_features_xd * num_heads * 2, gcn_linear_features)
        self.fc_g2_d2 = nn.Linear(gcn_linear_features, xd_output_dim)

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
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_gat_gcn2')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.early_stopping = EarlyStopping(checkpoint_path=checkpoint_path,
                                            patience=patience, delta=delta, verbose=self.verbose)
        self.criterion = nn.MSELoss()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        
    def _initialize_weights(self):
        # Initialize weights using He Initialization or Xavier, depending on the layer
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
        xd1 = F.dropout(xd1, p=self.dropout_gcn, training=self.training)
        xd1 = F.elu(self.gat1_d1(xd1, edge_index1))
        xd1 = F.dropout(xd1, p=self.dropout_gcn, training=self.training)
        xd1 = self.gcn1_d1(xd1, edge_index1)
        xd1 = self.relu(xd1)
        xd1 = torch.cat((gmp(xd1, batch_d1), gap(xd1, batch_d1)), dim=1)
        xd1 = self.fc_g1_d1(xd1)
        xd1 = self.relu(xd1)
        xd1 = self.dropout(xd1)
        xd1 = self.fc_g2_d2(xd1)
        xd1 = self.relu(xd1)
        
        # Feed graph network for Drug 2
        xd2, edge_index2, batch_d2 = data.xd2, data.edge_index2, data.batch_d2
        xd2 = F.dropout(xd2, p=self.dropout_gcn, training=self.training)
        xd2 = F.elu(self.gat1_d2(xd2, edge_index2))
        xd2 = F.dropout(xd2, p=self.dropout_gcn, training=self.training)
        xd2 = self.gcn1_d2(xd2, edge_index2)
        xd2 = self.relu(xd2)
        xd2 = torch.cat((gmp(xd2, batch_d2), gap(xd2, batch_d2)), dim=1)
        xd2 = self.fc_g1_d2(xd2)
        xd2 = self.relu(xd2)
        xd2 = self.dropout(xd2)
        xd2 = self.fc_g2_d2(xd2)
        xd2 = self.relu(xd2)

        # Cell line features and cell line target; Reshape them first as it's collated
        xc1, xc2, xc3, xtc = data.xc1, data.xc2, data.xc3, data.xtc
        xcl = self.cell_line_model(torch.cat((xc1, xc2, xc3, xtc), dim=1))
        
        # Concat processed cell line and drug features; proceed with FCN
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
    
    # Run whole training procedure
    def run_train(self, train_loader, val_loader, **kwargs):
        self.to(DEVICE)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                data = data.to(DEVICE)
                targets = data.labels

                self.optimizer.zero_grad()
                outputs = self(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Calculate average training loss for the epoch
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            self.scheduler.step()

            # Validate the model
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(DEVICE)
                    targets = data.labels
                    outputs = self(data)
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

    def perform_prediction(self, test_loader):
        self.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE)
                output = self(data)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, data.labels.cpu()), 0)
                    
        # Evaluate model prediction
        mae_prediction = mean_absolute_error(total_preds.numpy().flatten(), total_labels.numpy().flatten())
        mse_prediction = mean_squared_error(total_preds.numpy().flatten(), total_labels.numpy().flatten())
        logging.info(f"Mean Absolute Error is: {mae_prediction}")
        logging.info(f"Mean Squared Error is: {mse_prediction}")

        return total_preds.numpy().flatten()
