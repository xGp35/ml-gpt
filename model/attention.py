import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        
        # What does this mean - "Instantiation order matters for reproducible weights: key, query, value" . Is this specific to the problem evaluation? seems so
        self.C = embedding_dim
        self.d_k = attention_dim
        self.W_K = nn.Linear(self.C, self.d_k, bias = False)
        self.W_Q = nn.Linear(self.C, self.d_k, bias = False)
        self.W_V = nn.Linear(self.C, self.d_k, bias = False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded is - [B, T, C]
        # W_Q is stored as [A,C]. (out x in)
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        T = embedded.shape[1] 
        
        K = self.W_K(embedded) # K is [B, T , A]
        Q = self.W_Q(embedded) # Q is [B, T , A]
        V = self.W_V(embedded) # V is [B, T , A]

        attention_matrix = Q @ K.transpose(-2, -1) # [B, T, T]
        # Since Q and K are both [B, T, A], your Q @ K.T does not produce the intended shape. That could work is there was one sentence. For batched attention, we have to use ".transpose(-2,-1)". 
        attention_scores = attention_matrix / (self.d_k ** 0.5)

        mask = torch.tril(torch.ones(T, T))
        mask = (mask == 0) # Need to invert the mask so that the 1's in the matrix can be filled with -inf, when using masked_fill on attention_scores
        masked_scores = attention_scores.masked_fill(mask, value = float('-inf'))
        
        A = F.softmax(masked_scores, dim = -1) # softmax will be applied across last dimension in [B, T, T].
        return torch.round(A @ V, decimals=4)

