import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool
from .Cell_line import *

class GCNNet(torch.nn.Module):
    def __init__(self,  input_dim_drug, dropout_drug, dropout_final, 
                 dropout_celline,  dim_hidden = 128, output_drug = 256,
               output_celline = 256, step_size=10, gamma=0.1, learning_rate = 1e-3, weight_decay = 1e-04): #Not sure what input_dim_drug is
        super(GCNNet, self).__init__()

        ## Activations and whatnot
        self.relu = nn.ReLU()
        self.dropout_drug = nn.Dropout(dropout_drug)
        self.dropout_cellline = nn.Dropout(dropout_celline)
        self.dropout_final = nn.dropout(dropout_final)

        ## Drug 1
        self.drug1_conv1 = GCNConv(input_dim_drug, dim_hidden)
        self.drug1_bn1 = nn.BatchNorm1d(dim_hidden)
        self.drug1_conv2 = GCNConv(dim_hidden, dim_hidden*2)
        self.drug1_bn2 = nn.BatchNorm1d(dim_hidden*2)
        self.drug1_conv3 = GCNConv(dim_hidden*2, dim_hidden*4)
        self.drug1_bn3 = nn.BatchNorm1d(dim_hidden* 4)

        ## Dense Layers
        self.drug1_fc1 = torch.nn.Linear(dim_hidden*4, dim_hidden*2)
        self.drug1_fc2 = torch.nn.Linear(dim_hidden*2, output_drug)

        ## Drug 2
        self.drug2_conv1 = GCNConv(input_dim_drug, dim_hidden)
        self.drug2_bn1 = nn.BatchNorm1d(dim_hidden)
        self.drug2_conv2 = GCNConv(dim_hidden, dim_hidden*2)
        self.drug2_bn2 = nn.BatchNorm1d(dim_hidden*2)
        self.drug2_conv3 = GCNConv(dim_hidden*2, dim_hidden*4)
        self.drug2_bn3 = nn.BatchNorm1d(dim_hidden* 4)

        ## Dense Layers
        self.drug2_fc1 = torch.nn.Linear(dim_hidden*4, dim_hidden*2)
        self.drug2_fc2 = torch.nn.Linear(dim_hidden*2, output_drug)

        ## Cell Line
        self.cell_line_features = CellLine(output_dim=output_celline)  ## Need to check how to combine these 2

        ## Fully Connected Concat Layers Fully Connected layer input size 2*Drug_embedding_size + Cell_line_embedding
        self.fc1_mlp = nn.Linear(2*output_drug+output_celline, 1024)
        self.fc2_mlp = nn.Linear(1024, 256)
        self.fc3_mlp = nn.Linear(256, 1) # Not sure if we are including multiple outputs

        # ## Residual Layers 
        # self.transform_residual = nn.Linear(1024, 256)

        ## Initialization
        self._initialise_weights()
        self.criterion = torch.nn.MSELoss()
        self.gamma = gamma
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def _initialize_weights(self):
      """
      Weight Initialization using Kaiming init with ReLu
      """
      for m in self.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
          nn.init.xavier_uniform_(m.weight)
          if m.bias is not None:
              nn.init.constant_(m.bias, 0)
        elif isinstance(m, GCNConv):
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
          if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0) 
    
    def _initialize_optimizer(self):
      self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _initialize_scheduler(self):
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def forward(self, data):
        x_d1, edge_index_d1, cellline, batch = data.x1, data.edge_index1, data.batch, data.cellline
        x_d2, edge_index_d2, = data.x2, data.edge_index2, data.batch

        ## Drug 1
        x_d1 = self.drug1_conv1(x_d1, edge_index_d1)
        x_d1 = self.drug1_bn1(x_d1)
        x_d1 = self.relu(x_d1)

        x_d1 = self.drug1_conv2(x_d1, edge_index_d1)
        x_d1 = self.drug1_bn2(x_d1)
        x_d1 = self.relu(x_d1)

        x_d1 = self.drug1_conv3(x_d1, edge_index_d1)
        x_d1 = self.drug1_bn3(x_d1)
        x_d1 = self.relu(x_d1)
        x_d1 = global_max_pool(x_d1, batch)  # global max pooling
        
        ## Drug 2
        x_d2 = self.drug2_conv1(x_d2, edge_index_d2)
        x_d2 = self.drug2_bn1(x_d2)
        x_d2 = self.relu(x_d2)

        x_d2 = self.drug2_conv2(x_d2, edge_index_d2)
        x_d2 = self.drug2_bn2(x_d2)
        x_d2 = self.relu(x_d2)

        x_d2 = self.drug2_conv3(x_d2, edge_index_d2)
        x_d2 = self.drug2_bn3(x_d2)
        x_d2 = self.relu(x_d2)
        x_d2 = global_max_pool(x_d2, batch)  # global max pooling

        ## Cell Line (Add graph functionality)
        cell_line_features = self.cell_line_features(cellline)

        ## Concat
        combined_features = torch.cat([x_d1, x_d2, cell_line_features], dim=1)

        ## Dense Layers
        x = self.fc1_mlp(combined_features)
        x = self.relu(x)
        x = self.dropout_final(x)
        x = self.fc2_mlp(x)
        x = self.relu(x)
        output = self.fc3_mlp(x)
        
        # ## Residual Connections 
        # residual = self.transform_residual(x)
        # x = self.dropout_final(x)
        # x = self.fc2_mlp(x)
        # x = self.relu(x + residual)
        #output = self.fc3_mlp(x)

        ## Standardising output with outcome classes
        sigmoid = torch.nn.Sigmoid()
        output = sigmoid(output)
        output_normalized = output*(200) - 100  # [0,1]*200 - 100 = [-100, 100]

        return output_normalized
    
    

    def train_model(self, train_loader, val_loader, epochs, learning_rate, device, patience=10):
        self.to(device)
        self._initialize_optimizer(learning_rate)
        self._initialize_scheduler() 

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            epoch_loss = 0

            for data in train_loader:
                data = data.to(device)
                self.optimizer.zero_grad()
                output = self.forward(data)
                loss = self.criterion(output, data.y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step() 

            avg_train_loss = epoch_loss / len(train_loader)
            val_loss = self.validate(val_loader, device)

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]}')

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping")  ##  Simple Early Stopping
                break

    def validate(self, val_loader, device):
        self.eval()  # Set the model to evaluation mode
        total_loss = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = self.forward(data)
                loss = self.criterion(output, data.y)
                total_loss += loss.item()

        return total_loss / len(val_loader)