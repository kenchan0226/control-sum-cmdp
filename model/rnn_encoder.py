import torch
import torch.nn as nn
from model.positional_encoding import PositionalEncoding


class RNNEncoder(nn.Module):
    """
    Base class for rnn encoder
    """
    def forward(self, src, src_lens, src_mask=None):
        raise NotImplementedError


class RNNEncoderBasic(RNNEncoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0, use_positional_encoding=False):
        super(RNNEncoderBasic, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        #self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding_layer = PositionalEncoding(embed_size, 500)

    def forward(self, src, src_lens, src_mask=None, position_ids=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """
        # Debug
        #if math.isnan(self.rnn.weight_hh_l0[0,0].item()):
        #    logging.info('nan encoder parameter')
        src_embed = self.embedding(src) # [batch, src_len, embed_size]
        # positional encoding
        if self.use_positional_encoding:
            src_embed = self.positional_encoding_layer(src_embed, position_ids)
        #src_embed_np = src_embed.detach().cpu().numpy()[:, 0, :]

        # sort src_embed according to its length
        batch_size = src.size(0)
        assert len(src_lens) == batch_size
        sort_ind = sorted(range(batch_size), key=lambda i: src_lens[i], reverse=True)
        src_lens_sorted = [src_lens[i] for i in sort_ind]
        src_embed = reorder_sequence(src_embed, sort_ind, batch_first=True)
        #src_embed_sorted_np = src_embed.detach().cpu().numpy()[:, 0, :]

        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens_sorted, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)

        # restore the order of memory_bank
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(batch_size)]
        memory_bank = reorder_sequence(memory_bank, reorder_ind, batch_first=True)
        encoder_final_state = reorder_gru_states(encoder_final_state, reorder_ind)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1,:,:], encoder_final_state[-2,:,:]), 1) # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :] # [batch, hidden_size]

        return memory_bank.contiguous(), encoder_last_layer_final_state


def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first, [B, T, D] if batch first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]
    device = sequence_emb.device
    order = torch.LongTensor(order)
    order = order.to(device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_


def reorder_gru_states(gru_state, order):
    """
    gru_state: [num_layer * num_directions, batch, hidden_size]
    order: list of sequence length
    """
    assert len(order) == gru_state.size(1)
    order = torch.LongTensor(order).to(gru_state.device)
    sorted_state = gru_state.index_select(index=order, dim=1)
    return sorted_state


def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states