import torch 
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
from .Cell_line import *
class GIN_drug(nn.Module):
  def __init__(self, input_dim_drug, dropout_drug, dropout_final, dropout_celline, 
               train_epsilon = False, dim_hidden = 128, output_drug = 256,
               output_celline = 256, step_size=10, gamma=0.1, learning_rate = 1e-3, weight_decay = 1e-04): 
    super(GIN_drug, self).__init__()
    
    self.relu = nn.ReLU()
    self.dropout_drug = nn.Dropout(dropout_drug)
    self.dropout_cellline = nn.Dropout(dropout_celline)
    self.dropout_final = nn.Dropout(dropout_final)

    ## DRUG 1
    self.ginconv1_drug1 = GINConv(  ## 4 GIN layers because graph size is relatively small
        nn.Sequential(nn.Linear(input_dim_drug, dim_hidden), # Not including input dim here because I am not sure what the sizes are
                   nn.BatchNorm1d(dim_hidden),
                   nn.ReLU(),
                   nn.Linear(dim_hidden, dim_hidden),
                   self.relu()), train_eps = train_epsilon)
    self.ginconv2_drug1 = GINConv(
        nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                   nn.BatchNorm1d(dim_hidden),
                   nn.ReLU(),
                   nn.Linear(dim_hidden, dim_hidden),
                   self.relu()), train_eps = train_epsilon)
    self.ginconv3_drug1 = GINConv(
        nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                   nn.BatchNorm1d(dim_hidden),
                   nn.ReLU(),
                   nn.Linear(dim_hidden, dim_hidden),
                   self.relu()), train_epsilon)
    
    self.ginconv4_drug1 = GINConv(
        nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                   nn.BatchNorm1d(dim_hidden),
                   nn.ReLU(),
                   nn.Linear(dim_hidden, dim_hidden),
                   self.relu()), train_epsilon)
    
    self.fc1_drug1 = nn.Linear(dim_hidden*4, dim_hidden*2)
    self.fc2_drug1 = nn.Linear(dim_hidden*2, output_drug)


    ## Drug 2
    self.ginconv1_drug2 = GINConv(  ## 4 GIN layers because graph size is relatively small
        nn.Sequential(nn.Linear(input_dim_drug, dim_hidden),
                   nn.BatchNorm1d(dim_hidden),
                   nn.ReLU(),
                   nn.Linear(dim_hidden, dim_hidden),
                   self.relu()), train_eps = train_epsilon)
    self.ginconv2_drug2 = GINConv(
        nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                   nn.BatchNorm1d(dim_hidden),
                   nn.ReLU(),
                   nn.Linear(dim_hidden, dim_hidden),
                   self.relu()), train_eps = train_epsilon)
    self.ginconv3_drug2 = GINConv(
        nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                   nn.BatchNorm1d(dim_hidden),
                   nn.ReLU(),
                   nn.Linear(dim_hidden, dim_hidden),
                   self.relu()), train_epsilon)
    
    self.ginconv4_drug2 = GINConv(
        nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                   nn.BatchNorm1d(dim_hidden),
                   nn.ReLU(),
                   nn.Linear(dim_hidden, dim_hidden),
                   self.relu()), train_epsilon)
    
    self.fc1_drug2 = nn.Linear(dim_hidden*4, dim_hidden*2)
    self.fc2_drug2 = nn.Linear(dim_hidden*2, output_drug)

    ## CellLine
    self.cell_line_features = CellLine(output_dim=output_celline)

    ## Fully Connected layer input size 2*Drug_embedding_size + Cell_line_embedding
    self.fc1_mlp = nn.Linear(2*output_drug+output_celline, 1024)
    self.fc2_mlp = nn.Linear(1024, 256)
    self.fc3_mlp = nn.Linear(256, 1) # Not sure if we are including multiple outputs
    
    ## Weights and other things:
    self._initialise_weights()
    self.criterion = torch.nn.MSELoss()
    self.gamma = gamma
    self.step_size = step_size
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

  def _initialize_optimizer(self,):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
  
  def _initialize_scheduler(self):
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)


  def _initialise_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):   
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, GINConv): 
            for sub_module in m.nn.modules(): ## GIN networks have Linear Layers inside of them  So might need to double check.
                if isinstance(sub_module, nn.Linear):
                    nn.init.kaiming_uniform_(sub_module.weight, nonlinearity='relu')
                if sub_module.bias is not None:
                    nn.init.constant_(sub_module.bias, 0)


  def forward(self, data):
    x_d1, edge_index_d1, cellline, batch = data.x1, data.edge_index1, data.batch, data.cellline
    x_d2, edge_index_d2, = data.x2, data.edge_index2, data.batch  ## Change to match. 
                                                                  ## Excluding Target Cell

    ## GIN for Drug 1
    x1_d1 = self.ginconv1(x_d1, edge_index_d1) 
    x2_d1 = self.ginconv2(x1_d1, edge_index_d1)
    x3_d1 = self.ginconv3(x2_d1, edge_index_d1)
    x4_d1 = self.ginconv4(x3_d1, edge_index_d1)
    ## Global pooling Drug 1
    x1_d1 = global_add_pool(x1_d1, batch)
    x2_d1 = global_add_pool(x2_d1, batch)
    x3_d1 = global_add_pool(x3_d1, batch) 
    x4_d1 = global_add_pool(x4_d1, batch)
    ## Concat-D1
    xallgin_d1 = torch.cat((x1_d1, x2_d1, x3_d1, x4_d1), dim = 1) # Combined graph level embedding from neighbours
    ## Dense Layers-D1 
    xgin_d1 = self.relu(self.drug_fc1(xallgin_d1))
    xgin_d1 = self.dropout_drug(xgin_d1)
    xgin_d1 = self.drug_fc2(xgin_d1)

    ## GIN for Drug 2
    x1_d2 = self.ginconv1(x_d2, edge_index_d2) 
    x2_d2 = self.ginconv2(x1_d2, edge_index_d2)
    x3_d2 = self.ginconv3(x2_d2, edge_index_d2)
    x4_d2 = self.ginconv4(x3_d2, edge_index_d2)
    ## Global Pooling Drug2
    x1_d2 = global_add_pool(x1_d2, batch)
    x2_d2 = global_add_pool(x2_d2, batch)
    x3_d2 = global_add_pool(x3_d2, batch) 
    x4_d2 = global_add_pool(x4_d2, batch)
    ## Concat-D2
    xallgin_d2 = torch.cat((x1_d2, x2_d2, x3_d2, x4_d2), dim = 1) # Combined graph level embedding from neighbours
    ## Dense Layers-D2 
    xgin_d2 = self.relu(self.drug_fc1(xallgin_d2))
    xgin_d2 = self.dropout_drug(xgin_d2)
    xgin_d2 = self.drug_fc2(xgin_d1)

    ## Cell line feature extraction
    x_cl = self.cell_line_features(cellline)

    ## Concatting all layers
    x_all = torch.cat((xgin_d1, xgin_d2, x_cl), dim=1) ## include dropout?

    ## Dense concatted layers
    x_all = self.relu(self.fc1_mlp(x_all))
    x_all = self.dropout(x_all)
    x_all = self.relu(self.fc2_mlp(x_all))
    x_all = self.dropout(x_all)
    output = self.fc3_mlp(x_all)

    ## Limiting output to -100 to 100
    sigmoid = torch.nn.Sigmoid()
    output = sigmoid(output)
    output_normalized = output*(200) - 100 
    return output_normalized
  
  def train_model(self, train_loader, val_loader, epochs, device, patience=5):
    self.to(device)
    self._initialize_optimizer()
    self._initialize_scheduler() 

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        self.train()  ## Set the model to training mode
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