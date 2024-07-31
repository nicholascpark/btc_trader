import torch.nn as nn
import torch
import numpy as np
from models.attention import SimpleAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_layers=1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Linear(input_dim, hidden_size)
        self.position_encoding = nn.Parameter(torch.randn(1000, 1, hidden_size))
        self.layers = nn.ModuleList([SimpleAttention(hidden_size) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(0, 1)  # (seq_len, batch_size, input_dim)
        x = self.embedding(x)  # (seq_len, batch_size, hidden_size)
        x = x + self.position_encoding[:x.size(0), :, :]
        
        for layer in self.layers:
            x = layer(x)
        
        output = self.fc(x)
        return output.squeeze()