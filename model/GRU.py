import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, hidden_units, dropout=0.1):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)

        self.final = nn.Sequential(
            nn.Linear(hidden_size, hidden_units*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_units*2),
            nn.Linear(hidden_units*2, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, num_classes),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x = torch.transpose(x,1,2)  # (N,H_in,L) > (N,L,H_in)

        # Initialize hidden states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device,dtype=x.dtype)
    
        # Forward pass through LSTM layers
        out, _ = self.gru(x, h0)
        
        # Take the hidden state from the last time step
        out =  out[:,-1,:]

        # Pass through the first linear layer
        out = self.final(out)

        return out

