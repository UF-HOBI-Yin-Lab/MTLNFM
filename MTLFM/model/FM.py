import torch
import pandas as pd
from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear

class FM(torch.nn.Module):


    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, features,mask,mask_value):
        """
        :param features: Long tensor of size ``(batch_size, num_fields)``
        :mask: list of index of numerical features
        :mask_value: list of scalar of numerical features
        """
        
        x = self.embedding(features)
        for i in range(len(mask_value)):
            for j in range(len(mask_value[0])):
                if pd.isnull(mask_value[i][j]):
                    x[i][mask[i][j]] = x[i][mask[i][j]] * mask_value[i][j]
                    
        x = self.linear(x) + self.fm(x)
        return torch.sigmoid(x.squeeze(1))
    
def dnn_layers(input_dim,hidden_layers,dropout,output = False):
    # mlp and fc layers
    layers = []
    for hidden_dim in hidden_layers:
        layers.append(torch.nn.Linear(input_dim , hidden_dim))
        layers.append(torch.nn.BatchNorm1d(hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(p=dropout))
        input_dim = hidden_dim
    if output:
        layers.append(torch.nn.Linear(input_dim, 1))
    return layers

class MTLNFM_2task(torch.nn.Module):


    def __init__(self, field_dims, embed_dim,hidden_layers,dropout):
        super().__init__()
        self.max_dim = field_dims
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropout[0])
        )
        self.shared_layers = torch.nn.Sequential(*dnn_layers(embed_dim,hidden_layers,dropout[1]))
        
        self.linear1 = FeaturesLinear(embed_dim)
        self.linear2 = FeaturesLinear(embed_dim)
        self.fc1 = torch.nn.Linear(hidden_layers[-1], 1)
        self.fc2 = torch.nn.Linear(hidden_layers[-1], 1)
        
    def freeze_shared(self):
        self.fm.requires_grad_(False)
        self.shared_layers.requires_grad_(False) 
        
    def separate_training(self,task_selection):
        self.fc1.requires_grad_(task_selection[0])
        self.linear1.requires_grad_(task_selection[1])
        self.fc2.requires_grad_(task_selection[2])
        self.linear2.requires_grad_(task_selection[3])
            
    def forward(self, features,mask,mask_value):
        """
        :param features: Long tensor of size ``(batch_size, num_fields)``
        :mask: list of index of numerical features
        :mask_value: list of scalar of numerical features
        """
        
        x = self.embedding(features)
        for i in range(len(mask_value)):
            for j in range(len(mask_value[0])):
                if pd.isnull(mask_value[i][j]):
                    x[i][mask[i][j]] = x[i][mask[i][j]] * mask_value[i][j]

        x1 = self.linear1(x) + self.fc1(self.shared_layers(self.fm(x)))
        x2 = self.linear2(x) + self.fc2(self.shared_layers(self.fm(x)))

        return torch.sigmoid(x1.squeeze(1)),torch.sigmoid(x2.squeeze(1))

class MTLNFM(torch.nn.Module):


    def __init__(self, field_dims, embed_dim,hidden_layers,dropout):
        super().__init__()
        self.max_dim = field_dims
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropout[0])
        )
        self.shared_layers = torch.nn.Sequential(*dnn_layers(embed_dim,hidden_layers,dropout[1]))
        
        self.linear1 = FeaturesLinear(embed_dim)
        self.linear2 = FeaturesLinear(embed_dim)
        self.linear3 = FeaturesLinear(embed_dim)
        self.fc1 = torch.nn.Linear(hidden_layers[-1], 1)
        self.fc2 = torch.nn.Linear(hidden_layers[-1], 1)
        self.fc3 = torch.nn.Linear(hidden_layers[-1], 1)
        
    def freeze_shared(self):
        self.fm.requires_grad_(False)
        self.shared_layers.requires_grad_(False) 
        
    def separate_training(self,task_selection):
        self.fc1.requires_grad_(task_selection[0])
        self.linear1.requires_grad_(task_selection[1])
        self.fc2.requires_grad_(task_selection[2])
        self.linear2.requires_grad_(task_selection[3])
        self.fc3.requires_grad_(task_selection[4])
        self.linear3.requires_grad_(task_selection[5])
            
    def forward(self, features,mask,mask_value):
        """
        :param features: Long tensor of size ``(batch_size, num_fields)``
        :mask: list of index of numerical features
        :mask_value: list of scalar of numerical features
        """

        x = self.embedding(features)
        for i in range(len(mask_value)):
            for j in range(len(mask_value[0])):
                if pd.isnull(mask_value[i][j]):
                    x[i][mask[i][j]] = x[i][mask[i][j]] * mask_value[i][j]
        x1 = self.linear1(x) + self.fc1(self.shared_layers(self.fm(x)))
        x2 = self.linear2(x) + self.fc2(self.shared_layers(self.fm(x)))
        x3 = self.linear3(x) + self.fc3(self.shared_layers(self.fm(x)))

        return torch.sigmoid(x1.squeeze(1)),torch.sigmoid(x2.squeeze(1)),torch.sigmoid(x3.squeeze(1))
    
    
class STLNFM(torch.nn.Module):


    def __init__(self, field_dims, embed_dim,hidden_layers,dropout):
        super().__init__()
        self.max_dim = field_dims
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropout[0])
        )
        self.shared_layers = torch.nn.Sequential(*dnn_layers(embed_dim,hidden_layers,dropout[1]))
        
        self.linear1 = FeaturesLinear(embed_dim)
        self.fc1 = torch.nn.Linear(hidden_layers[-1], 1)

    def freeze_shared(self):
        self.fm.requires_grad_(False)
        self.shared_layers.requires_grad_(False) 
                
    def forward(self, features,mask,mask_value):
        """
        :param features: Long tensor of size ``(batch_size, num_fields)``
        :mask: list of index of numerical features
        :mask_value: list of scalar of numerical features
        """
        
        x = self.embedding(features)
        for i in range(len(mask_value)):
            for j in range(len(mask_value[0])):
                if pd.isnull(mask_value[i][j]):
                    x[i][mask[i][j]] = x[i][mask[i][j]] * mask_value[i][j]
                    
        x1 = self.linear1(x) + self.fc1(self.shared_layers(self.fm(x)))

        return torch.sigmoid(x1.squeeze(1))