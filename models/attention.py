import torch.nn as nn
import torch

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, hidden_size)
        attn_weights = torch.softmax(self.attention(x), dim=0)
        context = torch.sum(x * attn_weights, dim=0)
        return context
