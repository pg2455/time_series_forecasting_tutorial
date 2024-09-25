### Solution for LSTNet rough model building exercise
seq_length = 50
dataset = utils.SlidingFixedWindow(train_data, seq_length)
train_loader = DataLoader(dataset, batch_size=2,  shuffle=True)

inputs, targets = next(iter(train_loader))

print("Inputs shape", inputs.shape)
dropout = 0
n_out_channels = 10
input_dim = inputs.shape[-1]
window=2
bs = inputs.shape[0]
out_features = targets.shape[-1]

# Eq. (1) Convolutional Component
conv1 = nn.Conv2d(
        in_channels=1, 
        out_channels=n_out_channels, 
        kernel_size=(window, input_dim)
    )

h_conv = conv1(inputs.unsqueeze(1)).squeeze(-1)
print("Conv output shape:", h_conv.shape)

# Eq. (2) Recurrent Component 
hidden_state_dims_GRU1 = 32
GRU1 = nn.GRU(
        input_size=n_out_channels,
        hidden_size=hidden_state_dims_GRU1,
        batch_first=True, 
        dropout=dropout,
    )

h_conv_in = h_conv.permute(0, 2, 1)
print("GRU input shape: ", h_conv_in.shape)
H_gru, h_gru = GRU1(h_conv_in)
h_gru = h_gru.squeeze(0)
print("GRU output shape:", h_gru.shape)

# Eq. (3) Recurrent-skip Component (GRU for every p hidden states)
skip = 4
hidden_state_dims_GRU2 = 16
GRU2 = nn.GRU(
    input_size=n_out_channels,
    hidden_size=hidden_state_dims_GRU2,
    batch_first=True,
    dropout=dropout,
)

seq_len = h_conv_in.shape[1] // skip # each sequence will have these many elements
n_seq = skip # there will be these many sequences
c = h_conv_in[:, -int(seq_len * n_seq):] # discard the states which can't fit in the window
c = c.view(bs, seq_len, n_seq, c.shape[-1]).contiguous() # stride every n_seq before switching index

# switch the dimensions and obtain the input for GRU
c = c.permute(0, 2, 1, 3).contiguous().view(bs*n_seq, seq_len, c.shape[-1])

print("These must be equal: ", c[1, :, 1], h_conv_in[0, 2::skip, 1])

_, s = GRU2(c)
print("GRU2 Output shape:", s.shape)

# Eq. (4) Recurrent Skip Component (concatenation)
r = torch.cat((h_gru, s.view(bs, -1)), 1)
linear1 = nn.Linear(hidden_state_dims_GRU1 + skip*hidden_state_dims_GRU2, out_features)
res = linear1(r)
print("r shape:", res.shape)

# (optional) Temporal Attention Layer (replacing the Recurrent Skip Component)
print("H_gru shape:", H_gru.shape)
attn_layer = nn.MultiheadAttention(embed_dim=hidden_state_dims_GRU1, num_heads=4, batch_first=True)
attn_out, attn_ws = attn_layer(query=H_gru[:, -1:], key=H_gru, value=H_gru)
print("attn out shape: ", attn_out.shape)
print("attn ws shape: ", attn_ws.shape)
r2 = torch.cat((h_gru, attn_out.squeeze(1)), 1)

linear1_attn = nn.Linear(hidden_state_dims_GRU1 + hidden_state_dims_GRU1, out_features)
res_attn = linear1_attn(r2)
print("res attn shape: ", res_attn.shape)

# Eq. (5) Autoregressive Component (scaling sensitivity)
linear2 = nn.Linear(seq_length, 1)
z = linear2(inputs.view(bs, out_features, -1)).squeeze(-1)

# Eq. (6) Final result
Y_t = res + z 
Y_t.shape

