import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.first_layer = nn.Linear(784,512) # (in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.last_layer = nn.Linear(512,10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        output_dist = self.sigmoid( self.last_layer( self.dropout( self.relu( self.first_layer( images)))))
        return torch.round(output_dist, decimals=4)
