
import torch
import torch.nn as nn

class CellLine(nn.Module):
  def __init__(self, dropout_cellline, output_dim = 256): ## Need to fix kernel_sizxe, num_filters and embeddins as well as pooledlength. Max can be removed
    super(CellLine, self).__init__()

    ## Dropout and activations
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout_cellline)

    ## Array 1: Transcript expression levels input size
    self.array1_conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=7, padding=0, stride=2) ## 23808 -> 11901
    self.array1_bn1 = nn.BatchNorm1d(num_features=5) 
    self.array1_pool1 = nn.MaxPool1d(kernel_size = 5, stride = 3)  ## (11901 - 5)/3 -> 3965
    self.array1_conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5, padding=0, stride = 2)  ## 3965 -> 1980
    self.array1_bn2 = nn.BatchNorm1d(num_features=10) 
    self.array1_pool2 = nn.MaxPool1d(kernel_size = 7, stride = 3) ## 1980 -> 657
    self.array1_conv3 = nn.Conv1d(in_channels=10, out_channels=15, kernel_size=3, padding=0, stride = 2)  ## 657 -> 327
    self.array1_bn3 = nn.BatchNorm1d(num_features=15) 
    self.array1_pool3 = nn.MaxPool1d(kernel_size = 7, stride = 2) ## 327 -> 108 (after flatten 2430)
    
    ## Array 2: Proteomic Protiens  
    self.array2_conv1 = nn.Conv1d(in_channels=1, out_channels=7, kernel_size= 5, padding=0, stride = 2) ## 3171 -> 1583
    self.array2_bn1 = nn.BatchNorm1d(num_features=7)
    self.array2_pool1 = nn.MaxPool1d(kernel_size = 3, stride = 3) ##n1583 -> 526
    self.array2_conv2 = nn.Conv1d(in_channels=7, out_channels=14, kernel_size=3, padding=0, stride = 2) ## 526 -> 262
    self.array2_bn2 = nn.BatchNorm1d(num_features=14)
    self.array2_pool2 = nn.MaxPool1d(kernel_size = 3, stride = 2) #129 (after flatten 1806) 
    self.array2_conv3 = nn.Conv1d(in_channels=14, out_channels=21, kernel_size=3, padding=0, stride = 2) ## 129 -> 63
    self.array2_bn3 = nn.BatchNorm1d(num_features=21)
    self.array2_pool3 = nn.MaxPool1d(kernel_size = 3, stride = 1) #63 -> 60 (after flatten 1323) 

    ## Array 3: mRNA information
    self.array3_conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=0, stride = 2) ## 627 -> 311
    self.array3_bn1 = nn.BatchNorm1d(num_features=8)
    self.array3_pool1 = nn.MaxPool1d(kernel_size = 3, stride = 2) ## 311 -> 154
    self.array3_conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=0, stride = 2) ## 154 -> 73
    self.array3_bn2 = nn.BatchNorm1d(num_features=16)
    self.array3_pool2 = nn.MaxPool1d(kernel_size = 3, stride = 1) ## 73 -> 70 (after flatten 1200)


    ## Dense Layers: output to be 256
    self.fc1 = nn.Linear(in_features = 2430 + 1323 + 1200, out_features = output_dim*4)
    self.out = nn.Linear(output_dim*4, output_dim)


  def forward(self, x): # this should be something else id it a py_geometric.data object then
    # array1, array2, array3 = data.array1, data.array2, data.array3 or however we are naming it
    array1, array2, array3, target_cell = x[:, :, :23808], x[:, :, 23808:26979], x[:, :, 26979:27606], x[:, :, 27606:] ## Not sure if target cell should be used 
    
    ## Gene Expressions
    x_gene = self.array1_conv1(array1)
    x_gene = self.array1_bn1(x_gene)
    x_gene = self.relu(x_gene)
    x_gene = self.array1_pool1(x_gene)
    x_gene = self.array1_conv2(x_gene)
    x_gene = self.array1_bn2(x_gene)
    x_gene = self.relu(x_gene)
    x_gene = self.array1_pool2(x_gene)
    x_gene = self.array1_conv3(x_gene)
    x_gene = self.array1_bn3(x_gene)
    x_gene = self.relu(x_gene)
    x_gene = self.array1_pool3(x_gene)
    x_gene = x_gene.view(x_gene.size(0), -1)
    print("x_gene: ", x_gene.shape)


    ## Proteomic Proteins
    x_pro = self.array2_conv1(array2)
    x_pro = self.array2_bn1(x_pro)
    x_pro = self.relu(x_pro)
    x_pro = self.array2_pool1(x_pro)
    x_pro = self.array2_conv2(x_pro)
    x_pro = self.array2_bn2(x_pro)
    x_pro = self.relu(x_pro)
    x_pro = self.array2_pool2(x_pro)
    x_pro = self.array2_conv3(x_pro)
    x_pro = self.array2_bn3(x_pro)
    x_pro = self.relu(x_pro)
    x_pro = self.array2_pool3(x_pro)
    x_pro = x_pro.view(x_pro.size(0), -1)
    print("x_pro: ", x_pro.shape)

    ## MicroRNA Features
    x_mrna = self.array3_conv1(array3)
    x_mrna = self.array3_bn1(x_mrna)
    x_mrna = self.relu(x_mrna)
    x_mrna = self.array3_pool1(x_mrna)
    x_mrna = self.array3_conv2(x_mrna)
    x_mrna = self.array3_bn2(x_mrna)
    x_mrna = self.relu(x_mrna)
    x_mrna = self.array3_pool2(x_mrna)
    x_mrna = x_mrna.view(x_mrna.size(0), -1)
    print("x_mrna: ", x_mrna.shape)
    
    ## Cat Features
    x_cat = torch.cat((x_gene, x_pro, x_mrna), dim = 1)

    ## Dense Layers
    x = self.relu(self.fc1(x_cat))
    x = self.out(x)

    return x
