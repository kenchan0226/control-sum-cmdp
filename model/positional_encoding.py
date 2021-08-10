# adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)  # [1, max_len, d_model]
        self.d_model = d_model
        self.max_len = max_len
        #print("pe")
        #print(pe.cpu().numpy())

    def forward(self, x, position_ids=None):
        """
        :param x: [batch, seq_len, embed_dim]
        :param position_ids: [batch, seq_len]
        :return:
        """
        if position_ids is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            # index by position ids
            batch_size, seq_len, _ = x.size()
            pe_expanded = self.pe.expand(batch_size, self.max_len, self.d_model)
            position_ids = position_ids.unsqueeze(2)
            position_ids_expanded = position_ids.expand(batch_size, seq_len, self.d_model)
            pe_expanded = pe_expanded.gather(1, position_ids_expanded)
            x = x + pe_expanded
        return x

if __name__ == "__main__":
    positional_encoding_layer = PositionalEncoding(4, 10)
    x = torch.zeros(3, 7, 4)
    y = positional_encoding_layer(x)
    print("y")
    print(y.cpu().numpy())
    position_ids = torch.LongTensor([[5,6,7,1,2,3,4], [0,1,2,3,4,5,6], [3,4,1,2,3,4,5]])
    z = positional_encoding_layer(x, position_ids)
    print("z")
    print(z.cpu().numpy())
