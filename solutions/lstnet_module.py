### Solution for LSTNet Module
class LSTNet(nn.Module):
    def __init__(self, 
                 input_dim, 
                 out_features, 
                 seq_length, 
                 num_attn_heads = 4, hidden_state_dims_attn=32,
                 n_out_channels=10, window_size=2, hidden_state_dims_GRU1=32, skip=4, 
                 hidden_state_dims_GRU2=32, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
                in_channels=1, 
                out_channels=n_out_channels, 
                kernel_size=(window_size, input_dim)
            )
        
        self.GRU1 = nn.GRU(
            input_size=n_out_channels,
            hidden_size=hidden_state_dims_GRU1,
            batch_first=True, 
            dropout=dropout,
        )

        self.GRU2 = nn.GRU(
            input_size=n_out_channels,
            hidden_size=hidden_state_dims_GRU2,
            batch_first=True,
            dropout=dropout,
        )

        self.skip = skip
        self.linear1 = nn.Linear(hidden_state_dims_GRU1 + skip*hidden_state_dims_GRU2, out_features)
        self.linear2 = nn.Linear(seq_length, 1)   

        self.attn_layer = nn.MultiheadAttention(embed_dim=hidden_state_dims_attn, 
                                                num_heads=num_attn_heads,
                                                dropout=dropout, 
                                                batch_first=True)     
        self.linear1_attn = nn.Linear(hidden_state_dims_attn + hidden_state_dims_GRU1, out_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        
        batch_size = inputs.shape[0] 

        # Eq. (1) Convolutional Component
        h_conv = F.relu(self.conv1(inputs.unsqueeze(1)).squeeze(-1))
        h_conv = self.dropout(h_conv)
        
        # Eq. (2) Recurrent Component
        x_gru = h_conv.permute(0, 2, 1)
        H_gru, h_gru = self.GRU1(x_gru)
        h_gru = h_gru.squeeze(0)
        H_gru, h_gru = self.dropout(H_gru), self.dropout(h_gru)

        if self.skip > 0:
            # Eq. (3) Recurrent-skip Component (GRU for every p hidden states)        
            seq_len = x_gru.shape[1] // self.skip # each sequence will have these many elements
            n_seq = self.skip # there will be these many sequences
            c = x_gru[:, -int(seq_len * n_seq):] # discard the states which can't fit in the window
            c = c.view(batch_size, seq_len, n_seq, c.shape[-1]).contiguous() # stride every n_seq before switching index
            c = c.permute(0, 2, 1, 3).contiguous().view(batch_size*n_seq, seq_len, c.shape[-1])  # switch the dimensions and obtain the input for GRU
            _, s = self.GRU2(c)
            s = self.dropout(s)

            # Eq. (4) Recurrent Skip Component (concatenation)
            r = torch.cat((h_gru, s.view(batch_size, -1)), 1)
            res = self.linear1(r)
        else:
            attn_out, _ = self.attn_layer(query=H_gru[:, -1:], key=H_gru, value=H_gru)
            r2 = torch.cat((h_gru, attn_out.squeeze(1)), 1)
            res = self.linear1_attn(r2)

        # Eq. (5) Autoregressive Component (scaling sensitivity)
        z = self.linear2(inputs.view(batch_size, out_features, -1)).squeeze(-1)

        # Eq. (6) Final result
        Y_t = res + z

        return Y_t
