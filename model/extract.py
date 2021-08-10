import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .extract_util import sequence_mean, len_mask
from .extract_attention import prob_normalize

INI = 1e-2
MAX_EXT = 6

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    :param logits: [*, n_class]
    :return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)  # one-hot vector
    return (y_hard - y).detach() + y, ind.tolist()


class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None
        self._output_size = 3 * n_hidden

    def forward(self, input_):
        # input_:  [num_sents, seq_len]
        batch_size, seq_len = input_.size()
        emb_input = self._embedding(input_) #  [num_sents, seq_len, embed_size]
        # padding for CNN
        if seq_len <= 2:
            left_padding = self._embedding(input_.new_zeros(batch_size, 2))  # [batch, 2, embed_size]
            right_padding = self._embedding(input_.new_zeros(batch_size, 2))  # [batch, 2, embed_size]
            padded_emb_input = torch.cat([left_padding, emb_input, right_padding], dim=1)  # [batch, seq_len+4, embed_size]
        elif seq_len <= 4:
            left_padding = self._embedding(input_.new_zeros(batch_size, 1))  # [batch, 1, embed_size]
            right_padding = self._embedding(input_.new_zeros(batch_size, 1))  # [batch, 1, embed_size]
            padded_emb_input = torch.cat([left_padding, emb_input, right_padding],
                                         dim=1)  # [batch, seq_len+2, embed_size]
        else:
            padded_emb_input = emb_input

        conv_in = F.dropout(padded_emb_input.transpose(1, 2),
                            self._dropout, training=self.training)  # [batch, embed_size, seq_len]
        try:
            output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        except:
            print(input_.size())
            print(emb_input.size())
            print(conv_in.size())
            exit()
        return output, emb_input

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

    @property
    def output_size(self):
        return self._output_size


class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, max_ext_num=6):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop
        self._stop = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._stop, -INI, INI)

        self.max_ext_num = max_ext_num

    def forward(self, attn_mem, mem_sizes, temperature):
        """
        attn_mem: Tensor of size [batch_size, max_sent_num, input_dim]
        mem_sizes: a list contains the number of sentences in each sample
        """
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)  # init_i [batch, 1, input_dim]
        lstm_in = init_i.transpose(0, 1)  # lstm_in [1, batch, input_dim]
        all_ext_one_hot = []
        all_ext_inds = []
        num_extracted_sent = 0
        batch_size = attn_mem.size(0)
        finished_flag = [0] * batch_size
        while True:
            query, next_states = self._lstm(lstm_in, lstm_states)
            # query: [1, batch, hidden_dim]
            query = query.transpose(0, 1)  # [batch, 1, hidden_dim]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(attn_feat, query, self._attn_v, self._attn_wq)  # [batch, 1, Ns]
            score = score.squeeze(1)  # [batch, Ns]
            # gumbel softmax
            ext_one_hot, ext_inds = gumbel_softmax(score, temperature=temperature)  # one-hot vectors with size [batch, Ns]
            # append results
            all_ext_one_hot.append(ext_one_hot)
            all_ext_inds.append(ext_inds)
            num_extracted_sent += 1
            # update finish flag and check if terminates
            for batch_i, (ext, mem_size) in enumerate(zip(ext_inds, mem_sizes)):
                if ext == mem_size - 1:
                    finished_flag[batch_i] = 1
            if all(finished_flag) or num_extracted_sent == self.max_ext_num:
                break

            lstm_in = torch.bmm(ext_one_hot.unsqueeze(1), attn_mem)  # [batch, 1, input_dim]
            lstm_in = lstm_in.transpose(0, 1)  # [1, batch, input_dim]
            lstm_states = next_states
        return all_ext_one_hot, all_ext_inds

    def extract(self, attn_mem, mem_sizes, k, disable_selected_mask=False):
        """extract k sentences, decode only, batch_size==1"""
        # atten_mem: Tensor of size [1, max_sent_num, input_dim]
        if self._auto_stop:
            end_step = attn_mem.size(1)
            attn_mem = torch.cat([attn_mem, self._stop.view(1, 1, -1)], dim=1)  # [1, max_sent_num+1, input_dim]

        use_selected_mask = not disable_selected_mask
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        num_extracted_sent = 0
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)  # [1, 1, Ns]
            score = score.squeeze()  # [Ns]
            # set logit to -inf if the sentence is selected before
            if use_selected_mask:
                for e in extracts:
                    score[e] = -1e6
            ext = score.max(dim=0)[1].item()
            if self._auto_stop:  # break the loop if eos is selected, does not include eos to the extracts
                if ext == end_step:
                    break
            extracts.append(ext)
            num_extracted_sent += 1
            if (not self._auto_stop and num_extracted_sent == k) or num_extracted_sent == MAX_EXT:
                break
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output


class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0, auto_stop=False):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop, auto_stop
        )

    def forward(self, article_sents, sent_nums, temperature):
        enc_out, emb_inputs = self._encode(article_sents, sent_nums)
        # emb_inputs, len=batch_size, each item with [number_sents, seq_len, embed_size]
        # enc_out: [batch, max_n, hidden_dim]
        emb_input_tensor = torch.stack(emb_inputs, dim=0)  # [batch, Ns, seq_len, embed_size]
        batch_size = len(sent_nums)

        all_ext_one_hot, all_ext_inds = self._extractor(enc_out, sent_nums, temperature)
        # all_ext_one_hot, len=T, each item [batch, Ns]
        # TODO: select word embeddings
        for ext_one_hot in all_ext_one_hot:
            # ext_one_hot  [batch, Ns]
            ext_one_hot = ext_one_hot.view(batch_size, 1, 1, -1)
            torch.bmm(ext_one_hot, emb_input_tensor)  # [batch, ]

        return output

    def extract(self, article_sents, sent_nums=None, k=4, disable_selected_mask=False):
        enc_out, emb_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k, disable_selected_mask)
        return output

    def _encode(self, article_sents, sent_nums):
        """
        :param article_sents:
        :param sent_nums: a list of sent_nums for each article
        :return:
        """
        max_n = max(sent_nums)
        enc_sents = []
        emb_inputs = []
        for art_sent in article_sents:
            enc_sent, emb_input = self._sent_enc(art_sent)
            # emb_input: [num_sents, seq_len, embed_size]
            enc_sents.append(enc_sent) # each item has dimension [num_sents, 3*conv_size]
            emb_inputs.append(emb_input)
        def zero(n, device):
            z = torch.zeros(n, self._art_enc.input_size).to(device)
            return z

        # pad the article that with sent_num less than max_n
        enc_sent = torch.stack(
            [torch.cat([s, zero(max_n-n, s.device)], dim=0)
               if n != max_n
             else s
             for s, n in zip(enc_sents, sent_nums)],
            dim=0
        )
        lstm_out = self._art_enc(enc_sent, sent_nums)  # [batch, max_n, hidden_dim]
        return lstm_out, emb_inputs

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)  # [num_sents * num_]

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional




