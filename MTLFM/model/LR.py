import torch
from layer import FeaturesLinear,FeaturesEmbedding
import pandas as pd
class LR(torch.nn.Module):

    def __init__(self, field_dims,embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(embed_dim)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        
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
        x= x.to(features.get_device())
        x = self.linear(x)
        return torch.sigmoid(x.squeeze(1))
