import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16,1) # In Pytoch model, we don't include information about the batch size. So no B
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        averaged_sentences = torch.mean(self.embedding(x), dim = 1)
        return torch.round(self.sigmoid(self.linear_layer(averaged_sentences)), decimals=4)
