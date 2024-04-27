import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p, num_layers, is_bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout_p, bidirectional=is_bidirectional)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_size * 2 if is_bidirectional else hidden_size, hidden_size)

    def forward(self, input_seq):
        rnn_output, (hidden_state, cell_state) = self.rnn(input_seq)
        hidden_state = self.dropout(hidden_state[-2:, :, :]) if self.rnn.bidirectional else self.dropout(hidden_state)
        combined = torch.tanh(self.linear(hidden_state[-1]))
        return combined, hidden_state[-1]


class ModalityDecoder(nn.Module):
    def __init__(self, output_dim, hid_dim, dropout, depth, attention, bidirectional):
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.attention = attention
        self.rnn = nn.LSTM(output_dim+hid_dim, hid_dim, num_layers=depth, dropout=dropout, bidirectional=bidirectional)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_dec, s, j):
        input_dec = input_dec.unsqueeze(1).transpose(0, 1) 
        att = self.attention(s, j).unsqueeze(1)
        j = j.transpose(0, 1)
        context = torch.bmm(att, j).transpose(0, 1)
        rnn_input = torch.cat((input_dec, context), dim=2)
        output, (hidden, _) = self.rnn(rnn_input) 

        if self.bidirectional:
            output = torch.add(output[:,:,:hidden.shape[-1]], output[:,:,hidden.shape[-1]:])
            hidden = torch.add(hidden[-1], hidden[-2])

        input_dec = input_dec.squeeze(0)
        output = output.squeeze(0)
        context = context.squeeze(0)
        
        pred = self.fc_out(torch.cat((output, context), dim=1))
        
        return pred, hidden.squeeze(0)

