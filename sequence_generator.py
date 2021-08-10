"""
Adapted from
OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
and seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""

import sys
import torch
from beam import Beam
from beam import GNMTGlobalScorer

EPS = 1e-8

class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self,
                 model,
                 eos_idx,
                 bos_idx,
                 pad_idx,
                 beam_size,
                 max_sequence_length,
                 include_attn_dist=True,
                 length_penalty_factor=0.0,
                 coverage_penalty_factor=0.0,
                 length_penalty='none',
                 coverage_penalty='none',
                 cuda=True,
                 n_best=None,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 len_idx=-1
                 ):
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, dec_hidden) and outputs len(vocab) values
          eos_idx: the idx of the <eos> token
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
          coverage_attn: use coverage attention or not
          include_attn_dist: include the attention distribution in the sequence obj or not.
          length_normalization_factor: If != 0, a number x such that sequences are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of sequences depending on their lengths. For example, if
            x > 0 then longer sequences will be favored.
            alpha in: https://arxiv.org/abs/1609.08144
          length_normalization_const: 5 in https://arxiv.org/abs/1609.08144
        """
        self.model = model
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.len_idx = len_idx
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_penalty_factor = length_penalty_factor
        self.coverage_penalty_factor = coverage_penalty_factor
        self.include_attn_dist = include_attn_dist
        self.coverage_penalty = coverage_penalty
        self.global_scorer = GNMTGlobalScorer(length_penalty_factor, coverage_penalty_factor, coverage_penalty, length_penalty)
        self.cuda = cuda
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        if n_best is None:
            self.n_best = self.beam_size
        else:
            self.n_best = n_best

    def beam_search(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param word2idx: a dictionary
        """
        self.model.eval()
        batch_size = src.size(0)
        beam_size = self.beam_size
        max_src_len = src.size(1)

        # Encoding
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens, src_mask)
        # [batch_size, max_src_len, memory_bank_size], [batch_size, memory_bank_size]

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Init decoder state
        decoder_init_state = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        # init initial_input to be BOS token
        #decoder_init_input = src.new_ones((batch_size * beam_size, 1)) * self.bos_idx  # [batch_size*beam_size, 1]

        if self.model.coverage_attn:  # init coverage
            #coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, src_len]
            coverage = src.new_zeros((batch_size * beam_size, max_src_len), dtype=torch.float)  # [batch_size * beam_size, max_src_len]
        else:
            coverage = None

        if self.model.review_attn:
            decoder_memory_bank = decoder_init_state[-1, :, :].unsqueeze(1)  # [batch, 1, decoder_size]
            decoder_memory_bank = decoder_memory_bank.repeat(beam_size, 1, 1)
            assert decoder_memory_bank.size() == torch.Size([batch_size * beam_size, 1, self.model.decoder_size])
        else:
            decoder_memory_bank = None

        # expand memory_bank, src_mask
        memory_bank = memory_bank.repeat(beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
        src_mask = src_mask.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]
        src_oov = src_oov.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
        decoder_state = decoder_init_state.repeat(1, self.beam_size, 1)  # [dec_layers, batch_size * beam_size, decoder_size]

        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([word2idx[t]
                                for t in self.ignore_when_blocking])

        beam_list = [Beam(beam_size, n_best=self.n_best, cuda=self.cuda, global_scorer=self.global_scorer, pad=self.pad_idx, eos=self.eos_idx, bos=self.bos_idx, max_eos_per_output_seq=1, block_ngram_repeat=self.block_ngram_repeat, exclusion_tokens=exclusion_tokens) for _ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a, requires_grad=False)

        '''
        Run beam search.
        '''
        for t in range(1, self.max_sequence_length + 1):
            if all((b.done() for b in beam_list)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # b.get_current_tokens(): [beam_size]
            # torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
            # after transpose -> [beam, batch]
            # After flatten, it becomes
            # [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
            # this match the dimension of hidden state
            decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
                      .t().contiguous().view(-1))
            # decoder_input: [batch_size * beam_size]

            # Turn any copied words to UNKS
            if self.model.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # Convert the generated eos token to bos token, only useful in one2many_mode=2 or one2many_mode=3
            #decoder_input = decoder_input.masked_fill(decoder_input == self.eos_idx, self.bos_idx)


            #goal_vector = None

            # run one step of decoding
            # [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size], [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
            decoder_dist, decoder_state, context, attn_dist, _, coverage = \
                self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, max_num_oov, src_oov, coverage, decoder_memory_bank)
            log_decoder_dist = torch.log(decoder_dist + EPS)

            if self.model.review_attn:
                decoder_memory_bank = torch.cat([decoder_memory_bank, decoder_state[-1, :, :].unsqueeze(1)], dim=1)  # [batch_size * beam_size, t+1, decoder_size]

            # Compute a vector of batch x beam word scores
            log_decoder_dist = log_decoder_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, vocab_size]
            attn_dist = attn_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

            # Advance each beam
            for batch_idx, beam in enumerate(beam_list):
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])
                self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state, decoder_memory_bank)

        # Extract sentences from beam.
        result_dict = self._from_beam(beam_list)
        result_dict['batch_size'] = batch_size
        return result_dict

    def beam_search_diversity_attn(self, src, src_lens, query, query_lens, src_oov, src_mask, query_src_mask, oov_lists, word2idx):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param word2idx: a dictionary
        """
        self.model.eval()
        batch_size = src.size(0)
        beam_size = self.beam_size
        max_src_len = src.size(1)

        # Encoding
        # encode document
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens, src_mask)
        # [batch_size, max_src_len, memory_bank_size], [batch_size, memory_bank_size]

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # encode query
        query_memory_bank, _ = self.model.query_encoder(query, query_lens, query_src_mask)

        # Init decoder state and ctx
        decoder_init_state = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]
        context = self.model.init_context(memory_bank)  # [batch, memory_bank_size]

        # init initial_input to be BOS token
        #decoder_init_input = src.new_ones((batch_size * beam_size, 1)) * self.bos_idx  # [batch_size*beam_size, 1]

        if self.model.coverage_attn:  # init coverage
            #coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, src_len]
            coverage = src.new_zeros((batch_size * beam_size, max_src_len), dtype=torch.float)  # [batch_size * beam_size, max_src_len]
        else:
            coverage = None

        if self.model.review_attn:
            decoder_memory_bank = decoder_init_state[-1, :, :].unsqueeze(1)  # [batch, 1, decoder_size]
            decoder_memory_bank = decoder_memory_bank.repeat(beam_size, 1, 1)
            assert decoder_memory_bank.size() == torch.Size([batch_size * beam_size, 1, self.model.decoder_size])
        else:
            decoder_memory_bank = None

        # init orthogonal lstm init states
        if self.model.use_ortho_LSTM:
            ortho_lstm_h = self.model.decoder.init_h_t.view(1, -1).expand(batch_size * beam_size, -1)
            ortho_lstm_c = self.model.decoder.init_c_t.view(1, -1).expand(batch_size * beam_size, -1)
        else:
            ortho_lstm_h = None
            ortho_lstm_c = None

        # expand memory_bank, src_mask
        memory_bank = memory_bank.repeat(beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
        src_mask = src_mask.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]
        src_oov = src_oov.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
        decoder_state = decoder_init_state.repeat(1, self.beam_size, 1)  # [dec_layers, batch_size * beam_size, decoder_size]

        query_memory_bank = query_memory_bank.repeat(beam_size, 1, 1)
        query_src_mask = query_src_mask.repeat(beam_size, 1)

        context = context.repeat(beam_size, 1)

        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([word2idx[t]
                                for t in self.ignore_when_blocking])

        beam_list = [Beam(beam_size, n_best=self.n_best, cuda=self.cuda, global_scorer=self.global_scorer, pad=self.pad_idx, eos=self.eos_idx, bos=self.bos_idx, max_eos_per_output_seq=1, block_ngram_repeat=self.block_ngram_repeat, exclusion_tokens=exclusion_tokens) for _ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a, requires_grad=False)

        '''
        Run beam search.
        '''
        for t in range(1, self.max_sequence_length + 1):
            if all((b.done() for b in beam_list)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # b.get_current_tokens(): [beam_size]
            # torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
            # after transpose -> [beam, batch]
            # After flatten, it becomes
            # [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
            # this match the dimension of hidden state
            decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
                      .t().contiguous().view(-1))
            # decoder_input: [batch_size * beam_size]

            # Turn any copied words to UNKS
            if self.model.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # Convert the generated eos token to bos token, only useful in one2many_mode=2 or one2many_mode=3
            #decoder_input = decoder_input.masked_fill(decoder_input == self.eos_idx, self.bos_idx)


            #goal_vector = None

            # run one step of decoding
            # [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size], [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
            decoder_dist, decoder_state, context, attn_dist, _, coverage, ortho_lstm_h, ortho_lstm_c = \
                self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, query_memory_bank, query_src_mask, max_num_oov, src_oov, context, coverage, ortho_lstm_h, ortho_lstm_c, decoder_memory_bank)
            log_decoder_dist = torch.log(decoder_dist + EPS)

            if self.model.review_attn:
                decoder_memory_bank = torch.cat([decoder_memory_bank, decoder_state[-1, :, :].unsqueeze(1)], dim=1)  # [batch_size * beam_size, t+1, decoder_size]

            # Compute a vector of batch x beam word scores
            log_decoder_dist = log_decoder_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, vocab_size]
            attn_dist = attn_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

            # Advance each beam
            for batch_idx, beam in enumerate(beam_list):
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])
                self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state, decoder_memory_bank)

        # Extract sentences from beam.
        result_dict = self._from_beam(beam_list)
        result_dict['batch_size'] = batch_size
        return result_dict

    def beam_search_with_style(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx, style_label):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param word2idx: a dictionary
        :param style_label: a LongTensor, [batch]
        """
        self.model.eval()
        batch_size = src.size(0)
        beam_size = self.beam_size
        max_src_len = src.size(1)

        # Encoding
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens, src_mask)
        # [batch_size, max_src_len, memory_bank_size], [batch_size, memory_bank_size]

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Init decoder state
        decoder_init_state = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        # init initial_input to be BOS token
        #decoder_init_input = src.new_ones((batch_size * beam_size, 1)) * self.bos_idx  # [batch_size*beam_size, 1]

        if self.model.coverage_attn:  # init coverage
            #coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, src_len]
            coverage = src.new_zeros((batch_size * beam_size, max_src_len), dtype=torch.float)  # [batch_size * beam_size, max_src_len]
        else:
            coverage = None

        if self.model.review_attn:
            decoder_memory_bank = decoder_init_state[-1, :, :].unsqueeze(1)  # [batch, 1, decoder_size]
            decoder_memory_bank = decoder_memory_bank.repeat(beam_size, 1, 1)
            assert decoder_memory_bank.size() == torch.Size([batch_size * beam_size, 1, self.model.decoder_size])
        else:
            decoder_memory_bank = None

        # expand memory_bank, src_mask
        memory_bank = memory_bank.repeat(beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
        src_mask = src_mask.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]
        src_oov = src_oov.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
        decoder_state = decoder_init_state.repeat(1, self.beam_size, 1)  # [dec_layers, batch_size * beam_size, decoder_size]
        style_embedding = self.model.style_embedding(style_label).repeat(self.beam_size, 1)  # [batch * beam, embed_size]

        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([word2idx[t]
                                for t in self.ignore_when_blocking])

        beam_list = [Beam(beam_size, n_best=self.n_best, cuda=self.cuda, global_scorer=self.global_scorer, pad=self.pad_idx, eos=self.eos_idx, bos=self.bos_idx, max_eos_per_output_seq=1, block_ngram_repeat=self.block_ngram_repeat, exclusion_tokens=exclusion_tokens) for _ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a, requires_grad=False)

        '''
        Run beam search.
        '''
        for t in range(1, self.max_sequence_length + 1):
            if all((b.done() for b in beam_list)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # b.get_current_tokens(): [beam_size]
            # torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
            # after transpose -> [beam, batch]
            # After flatten, it becomes
            # [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
            # this match the dimension of hidden state
            decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
                      .t().contiguous().view(-1))
            # decoder_input: [batch_size * beam_size]

            # Turn any copied words to UNKS
            if self.model.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # Convert the generated eos token to bos token, only useful in one2many_mode=2 or one2many_mode=3
            #decoder_input = decoder_input.masked_fill(decoder_input == self.eos_idx, self.bos_idx)


            #goal_vector = None

            # run one step of decoding
            # [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size], [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
            decoder_dist, decoder_state, context, attn_dist, _, coverage = \
                self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, max_num_oov, src_oov, coverage, decoder_memory_bank, style_embedding)
            log_decoder_dist = torch.log(decoder_dist + EPS)

            if self.model.review_attn:
                decoder_memory_bank = torch.cat([decoder_memory_bank, decoder_state[-1, :, :].unsqueeze(1)], dim=1)  # [batch_size * beam_size, t+1, decoder_size]

            # Compute a vector of batch x beam word scores
            log_decoder_dist = log_decoder_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, vocab_size]
            attn_dist = attn_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

            # Advance each beam
            for batch_idx, beam in enumerate(beam_list):
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])
                self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state, decoder_memory_bank)

        # Extract sentences from beam.
        result_dict = self._from_beam(beam_list)
        result_dict['batch_size'] = batch_size
        return result_dict

    def beam_search_with_exact_len(self, src, src_lens, src_oov, src_mask, oov_lists, word2idx, exact_output_lens):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param word2idx: a dictionary
        :param exact_output_lens: a list of int
        """
        self.model.eval()
        batch_size = src.size(0)
        beam_size = self.beam_size
        max_src_len = src.size(1)

        # Encoding
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens, src_mask)
        # [batch_size, max_src_len, memory_bank_size], [batch_size, memory_bank_size]

        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Init decoder state
        decoder_init_state = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        # init initial_input to be BOS token
        #decoder_init_input = src.new_ones((batch_size * beam_size, 1)) * self.bos_idx  # [batch_size*beam_size, 1]

        if self.model.coverage_attn:  # init coverage
            #coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, src_len]
            coverage = src.new_zeros((batch_size * beam_size, max_src_len), dtype=torch.float)  # [batch_size * beam_size, max_src_len]
        else:
            coverage = None

        if self.model.review_attn:
            decoder_memory_bank = decoder_init_state[-1, :, :].unsqueeze(1)  # [batch, 1, decoder_size]
            decoder_memory_bank = decoder_memory_bank.repeat(beam_size, 1, 1)
            assert decoder_memory_bank.size() == torch.Size([batch_size * beam_size, 1, self.model.decoder_size])
        else:
            decoder_memory_bank = None

        # expand memory_bank, src_mask
        memory_bank = memory_bank.repeat(beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
        src_mask = src_mask.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]
        src_oov = src_oov.repeat(self.beam_size, 1)  # [batch * beam_size, src_seq_len]
        decoder_state = decoder_init_state.repeat(1, self.beam_size, 1)  # [dec_layers, batch_size * beam_size, decoder_size]

        len_idx_tensor = torch.LongTensor([self.len_idx] * batch_size).to(device=memory_bank.device)  # [batch_size]
        exact_len_embedding = self.model.decoder.embedding(len_idx_tensor)  # [batch_size, embed_size]
        # multiplied by length
        exact_output_lens_tensor = torch.FloatTensor(exact_output_lens).to(device=memory_bank.device)  # [batch_size]
        multiplied_exact_len_embedding = torch.bmm(exact_len_embedding.unsqueeze(2),
                                                   exact_output_lens_tensor.view(-1, 1, 1))
        multiplied_exact_len_embedding = multiplied_exact_len_embedding.squeeze()  # [batch_size, embed_size]
        multiplied_exact_len_embedding = multiplied_exact_len_embedding.repeat(self.beam_size, 1)  # [batch * beam, embed_size]

        #style_embedding = self.model.style_embedding(style_label).repeat(self.beam_size, 1)  # [batch * beam, embed_size]

        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([word2idx[t]
                                for t in self.ignore_when_blocking])

        beam_list = [Beam(beam_size, n_best=self.n_best, cuda=self.cuda, global_scorer=self.global_scorer, pad=self.pad_idx, eos=self.eos_idx, bos=self.bos_idx, max_eos_per_output_seq=1, block_ngram_repeat=self.block_ngram_repeat, exclusion_tokens=exclusion_tokens) for _ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a, requires_grad=False)

        '''
        Run beam search.
        '''
        for t in range(1, self.max_sequence_length + 1):
            if all((b.done() for b in beam_list)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            # b.get_current_tokens(): [beam_size]
            # torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
            # after transpose -> [beam, batch]
            # After flatten, it becomes
            # [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
            # this match the dimension of hidden state
            decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
                      .t().contiguous().view(-1))
            # decoder_input: [batch_size * beam_size]

            # Turn any copied words to UNKS
            if self.model.copy_attn:
                decoder_input = decoder_input.masked_fill(
                    decoder_input.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # Convert the generated eos token to bos token, only useful in one2many_mode=2 or one2many_mode=3
            #decoder_input = decoder_input.masked_fill(decoder_input == self.eos_idx, self.bos_idx)


            #goal_vector = None

            # run one step of decoding
            # [flattened_batch, vocab_size], [dec_layers, flattened_batch, decoder_size], [flattened_batch, memory_bank_size], [flattened_batch, src_len], [flattened_batch, src_len]
            decoder_dist, decoder_state, context, attn_dist, _, coverage = \
                self.model.decoder(decoder_input, decoder_state, memory_bank, src_mask, max_num_oov, src_oov, coverage, decoder_memory_bank, multiplied_exact_len_embedding)
            log_decoder_dist = torch.log(decoder_dist + EPS)

            if self.model.review_attn:
                decoder_memory_bank = torch.cat([decoder_memory_bank, decoder_state[-1, :, :].unsqueeze(1)], dim=1)  # [batch_size * beam_size, t+1, decoder_size]

            # Compute a vector of batch x beam word scores
            log_decoder_dist = log_decoder_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, vocab_size]
            attn_dist = attn_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

            # Advance each beam
            for batch_idx, beam in enumerate(beam_list):
                beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])
                self.beam_decoder_state_update(batch_idx, beam.get_current_origin(), decoder_state, decoder_memory_bank)

        # Extract sentences from beam.
        result_dict = self._from_beam(beam_list)
        result_dict['batch_size'] = batch_size
        return result_dict

    def _from_beam(self, beam_list):
        ret = {"predictions": [], "scores": [], "attention": []}
        for b in beam_list:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            # Collect all the decoded sentences in to hyps (list of list of idx) and attn (list of tensor)
            for i, (times, k) in enumerate(ks[:n_best]):
                # Get the corresponding decoded sentence, and also the attn dist [seq_len, memory_bank_size].
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)  # 3d list of idx (zero dim tensor), with len [batch_size, n_best, output_seq_len]
            ret['scores'].append(scores)  # a 2d list of zero dim tensor, with len [batch_size, n_best]
            ret["attention"].append(attn)  # a 2d list of FloatTensor[output sequence length, src_len] , with len [batch_size, n_best]
            # hyp[::-1]: a list of idx (zero dim tensor), with len = output sequence length
            # torch.stack(attn): FloatTensor, with size: [output sequence length, src_len]
        return ret

    def beam_decoder_state_update(self, batch_idx, beam_indices, decoder_state, decoder_memory_bank=None):
        """
        :param batch_idx: int
        :param beam_indices: a long tensor of previous beam indices, size: [beam_size]
        :param decoder_state: [dec_layers, flattened_batch_size, decoder_size]
        :return:
        """
        decoder_layers, flattened_batch_size, decoder_size = list(decoder_state.size())
        assert flattened_batch_size % self.beam_size == 0
        original_batch_size = flattened_batch_size//self.beam_size
        # select the hidden states of a particular batch, [dec_layers, batch_size * beam_size, decoder_size] -> [dec_layers, beam_size, decoder_size]
        decoder_state_transformed = decoder_state.view(decoder_layers, self.beam_size, original_batch_size, decoder_size)[:, :, batch_idx]
        # select the hidden states of the beams specified by the beam_indices -> [dec_layers, beam_size, decoder_size]
        decoder_state_transformed.data.copy_(decoder_state_transformed.data.index_select(1, beam_indices))

        if decoder_memory_bank is not None:
            # [batch_size * beam_size, t+1, decoder_size] -> [beam_size, t-1, decoder_size]
            decoder_memory_bank_transformed = decoder_memory_bank.view(self.beam_size, original_batch_size, -1, decoder_size)[:, batch_idx, :, :]
            # select the hidden states of the beams specified by the beam_indices -> [beam_size, t-1, decoder_size]
            decoder_memory_bank_transformed.data.copy_(decoder_memory_bank_transformed.data.index_select(0, beam_indices))

    def sample(self, src, src_lens, src_oov, src_mask, oov_lists, greedy=False, entropy_regularize=False, position_ids=None):
        # src, src_lens, src_oov, src_mask, oov_lists, word2idx
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param max_sample_length: The max length of sequence that can be sampled by the model
        :param greedy: whether to sample the word with max prob at each decoding step
        :return:
        """
        batch_size, max_src_len = list(src.size())
        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Encoding
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens, src_mask, position_ids)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.model.num_directions * self.model.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.model.num_directions * self.model.encoder_size])
        if greedy and entropy_regularize:
            raise ValueError("When using greedy, should not use entropy regularization.")

        # Init decoder state
        h_t_init = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        if self.model.coverage_attn:
            coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, max_src_seq]
        else:
            coverage = None

        if self.model.review_attn:
            decoder_memory_bank = h_t_init[-1, :, :].unsqueeze(1)  # [batch, 1, decoder_size]
            assert decoder_memory_bank.size() == torch.Size([batch_size, 1, self.model.decoder_size])
        else:
            decoder_memory_bank = None

        location_of_eos_for_each_batch = torch.zeros(batch_size, dtype=torch.long)

        # init y_t to be BOS token
        y_t_init = src.new_ones(batch_size) * self.bos_idx  # [batch_size]
        sample_list = [{"prediction": [], "attention": [], "done": False} for _ in range(batch_size)]
        log_selected_token_dist = []
        #prediction_all = src.new_ones(batch_size, max_sample_length) * self.pad_idx

        unfinished_mask = src.new_ones((batch_size, 1), dtype=torch.bool)  # all seqs in a batch are unfinished at the beginning
        unfinished_mask_all = [unfinished_mask]
        pred_counters = src.new_zeros(batch_size, dtype=torch.uint8)  # [batch_size]
        eos_idx_mask = y_t_init == self.eos_idx  # a vector [batch_size], 1 indicates location of eos at previous time step
        eos_idx_mask_all = [eos_idx_mask.unsqueeze(1)]

        if entropy_regularize:
            entropy = torch.zeros(batch_size).to(src.device)
        else:
            entropy = None

        for t in range(self.max_sequence_length):
            if t > 0:
                eos_idx_mask = (y_t_next == self.eos_idx)  # [batch_size]
                pred_counters += eos_idx_mask
                eos_idx_mask_all.append(eos_idx_mask.unsqueeze(1))
                unfinished_mask = pred_counters < 1
                unfinished_mask = unfinished_mask.unsqueeze(1)
                unfinished_mask_all.append(unfinished_mask)

            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next

            if self.model.review_attn:
                if t > 0:
                    decoder_memory_bank = torch.cat([decoder_memory_bank, h_t[-1, :, :].unsqueeze(1)], dim=1)  # [batch, t+1, decoder_size]

            # Turn any copied words to UNKS
            if self.model.copy_attn:
                y_t = y_t.masked_fill(
                    y_t.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # [batch, vocab_size], [dec_layers, batch, decoder_size], [batch, memory_bank_size], [batch, src_len], [batch, src_len]
            decoder_dist, h_t_next, context, attn_dist, _, coverage = \
                self.model.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage, decoder_memory_bank)

            log_decoder_dist = torch.log(decoder_dist + EPS)  # [batch, vocab_size]

            if entropy_regularize:
                entropy -= torch.bmm(decoder_dist.unsqueeze(1), log_decoder_dist.unsqueeze(2)).view(batch_size)  # [batch]

            if greedy:  # greedy decoding, only use in self-critical
                selected_token_dist, prediction = torch.max(decoder_dist, 1)
                selected_token_dist = selected_token_dist.unsqueeze(1)  # [batch, 1]
                prediction = prediction.unsqueeze(1)  # [batch, 1]
                log_selected_token_dist.append(torch.log(selected_token_dist + EPS))
            else:  # sampling according to the probability distribution from the decoder
                prediction = torch.multinomial(decoder_dist, 1)  # [batch, 1]
                # select the probability of sampled tokens, and then take log, size: [batch, 1], append to a list
                log_selected_token_dist.append(log_decoder_dist.gather(1, prediction))

            for batch_idx, sample in enumerate(sample_list):
                if not sample['done']:
                    sample['prediction'].append(prediction[batch_idx][0])  # 0 dim tensor
                    sample['attention'].append(attn_dist[batch_idx])  # [src_len] tensor
                    if int(prediction[batch_idx][0].item()) == self.model.eos_idx:
                        sample['done'] = True
                        location_of_eos_for_each_batch[batch_idx] = t
                else:
                    pass

            prediction = prediction * unfinished_mask.type_as(prediction)  # replace the input of terminated sequnece (not include eos) into 0 (PAD token)

            # prediction_all[:, t] = prediction[:, 0]
            y_t_next = prediction[:, 0]  # [batch]

            if all((s['done'] for s in sample_list)):
                break

        for sample in sample_list:
            sample['attention'] = torch.stack(sample['attention'], dim=0)  # [trg_len, src_len]

        log_selected_token_dist = torch.cat(log_selected_token_dist, dim=1)  # [batch, t]
        assert log_selected_token_dist.size() == torch.Size([batch_size, t+1])
        #output_mask = torch.ne(prediction_all, self.pad_idx)[:, :t+1]  # [batch, t]
        #output_mask = output_mask.type(torch.FloatTensor).to(src.device)

        unfinished_mask_all = torch.cat(unfinished_mask_all, dim=1).type_as(log_selected_token_dist)
        assert unfinished_mask_all.size() == log_selected_token_dist.size()
        #assert output_mask.size() == log_selected_token_dist.size()

        #pred_idx_all = torch.cat(pred_idx_all, dim=1).type(torch.LongTensor).to(src.device)
        #assert pred_idx_all.size() == log_selected_token_dist.size()

        eos_idx_mask_all = torch.cat(eos_idx_mask_all, dim=1).to(src.device)
        assert eos_idx_mask_all.size() == log_selected_token_dist.size()

        #return sample_list, log_selected_token_dist, unfinished_mask_all, pred_idx_all
        """
        if entropy_regularize:
            return sample_list, log_selected_token_dist, unfinished_mask_all, eos_idx_mask_all, entropy
        else:
            return sample_list, log_selected_token_dist, unfinished_mask_all, eos_idx_mask_all
        """
        return sample_list, log_selected_token_dist, unfinished_mask_all, eos_idx_mask_all, entropy, location_of_eos_for_each_batch

    def sample_with_exact_len_input(self, src, src_lens, src_oov, src_mask, oov_lists, exact_output_lens, greedy=False, entropy_regularize=False):
        # src, src_lens, src_oov, src_mask, oov_lists, word2idx
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param oov_lists: list of oov words (idx2word) for each batch, len=batch
        :param max_sample_length: The max length of sequence that can be sampled by the model
        :param greedy: whether to sample the word with max prob at each decoding step
        :return:
        """
        batch_size, max_src_len = list(src.size())
        max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

        # Encoding
        memory_bank, encoder_final_state = self.model.encoder(src, src_lens, src_mask)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.model.num_directions * self.model.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.model.num_directions * self.model.encoder_size])
        if greedy and entropy_regularize:
            raise ValueError("When using greedy, should not use entropy regularization.")

        # Init decoder state
        h_t_init = self.model.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]

        if self.model.coverage_attn:
            coverage = torch.zeros_like(src, dtype=torch.float)  # [batch, max_src_seq]
        else:
            coverage = None

        if self.model.review_attn:
            decoder_memory_bank = h_t_init[-1, :, :].unsqueeze(1)  # [batch, 1, decoder_size]
            assert decoder_memory_bank.size() == torch.Size([batch_size, 1, self.model.decoder_size])
        else:
            decoder_memory_bank = None

        location_of_eos_for_each_batch = torch.zeros(batch_size, dtype=torch.long)

        # init y_t to be BOS token
        y_t_init = src.new_ones(batch_size) * self.bos_idx  # [batch_size]

        len_idx_tensor = torch.LongTensor([self.len_idx] * batch_size).to(device=y_t_init.device)  # [batch_size]
        exact_len_embedding = self.model.decoder.embedding(len_idx_tensor)  # [batch_size, embed_size]
        # multiplied by length
        exact_output_lens_tensor = torch.FloatTensor(exact_output_lens).to(device=y_t_init.device)  # [batch_size]
        multiplied_exact_len_embedding = torch.bmm(exact_len_embedding.unsqueeze(2),
                                                   exact_output_lens_tensor.view(-1, 1, 1))
        multiplied_exact_len_embedding = multiplied_exact_len_embedding.squeeze()  # [batch_size, embed_size]

        sample_list = [{"prediction": [], "attention": [], "done": False} for _ in range(batch_size)]
        log_selected_token_dist = []

        #prediction_all = src.new_ones(batch_size, max_sample_length) * self.pad_idx

        unfinished_mask = src.new_ones((batch_size, 1), dtype=torch.bool)  # all seqs in a batch are unfinished at the beginning
        unfinished_mask_all = [unfinished_mask]
        pred_counters = src.new_zeros(batch_size, dtype=torch.uint8)  # [batch_size]
        eos_idx_mask = y_t_init == self.eos_idx  # a vector [batch_size], 1 indicates location of eos at previous time step
        eos_idx_mask_all = [eos_idx_mask.unsqueeze(1)]

        if entropy_regularize:
            entropy = torch.zeros(batch_size).to(src.device)
        else:
            entropy = None

        for t in range(self.max_sequence_length):
            if t > 0:
                eos_idx_mask = (y_t_next == self.eos_idx)  # [batch_size]
                pred_counters += eos_idx_mask
                eos_idx_mask_all.append(eos_idx_mask.unsqueeze(1))
                unfinished_mask = pred_counters < 1
                unfinished_mask = unfinished_mask.unsqueeze(1)
                unfinished_mask_all.append(unfinished_mask)

            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next

            if self.model.review_attn:
                if t > 0:
                    decoder_memory_bank = torch.cat([decoder_memory_bank, h_t[-1, :, :].unsqueeze(1)], dim=1)  # [batch, t+1, decoder_size]

            # Turn any copied words to UNKS
            if self.model.copy_attn:
                y_t = y_t.masked_fill(
                    y_t.gt(self.model.vocab_size - 1), self.model.unk_idx)

            # [batch, vocab_size], [dec_layers, batch, decoder_size], [batch, memory_bank_size], [batch, src_len], [batch, src_len]
            decoder_dist, h_t_next, context, attn_dist, _, coverage = \
                self.model.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage, decoder_memory_bank, multiplied_exact_len_embedding)

            log_decoder_dist = torch.log(decoder_dist + EPS)  # [batch, vocab_size]

            if entropy_regularize:
                entropy -= torch.bmm(decoder_dist.unsqueeze(1), log_decoder_dist.unsqueeze(2)).view(batch_size)  # [batch]

            if greedy:  # greedy decoding, only use in self-critical
                selected_token_dist, prediction = torch.max(decoder_dist, 1)
                selected_token_dist = selected_token_dist.unsqueeze(1)  # [batch, 1]
                prediction = prediction.unsqueeze(1)  # [batch, 1]
                log_selected_token_dist.append(torch.log(selected_token_dist + EPS))
            else:  # sampling according to the probability distribution from the decoder
                prediction = torch.multinomial(decoder_dist, 1)  # [batch, 1]
                # select the probability of sampled tokens, and then take log, size: [batch, 1], append to a list
                log_selected_token_dist.append(log_decoder_dist.gather(1, prediction))

            for batch_idx, sample in enumerate(sample_list):
                if not sample['done']:
                    sample['prediction'].append(prediction[batch_idx][0])  # 0 dim tensor
                    sample['attention'].append(attn_dist[batch_idx])  # [src_len] tensor
                    if int(prediction[batch_idx][0].item()) == self.model.eos_idx:
                        sample['done'] = True
                        location_of_eos_for_each_batch[batch_idx] = t
                else:
                    pass

            prediction = prediction * unfinished_mask.type_as(prediction)  # replace the input of terminated sequnece (not include eos) into 0 (PAD token)

            # prediction_all[:, t] = prediction[:, 0]
            y_t_next = prediction[:, 0]  # [batch]

            if all((s['done'] for s in sample_list)):
                break

        for sample in sample_list:
            sample['attention'] = torch.stack(sample['attention'], dim=0)  # [trg_len, src_len]

        log_selected_token_dist = torch.cat(log_selected_token_dist, dim=1)  # [batch, t]
        assert log_selected_token_dist.size() == torch.Size([batch_size, t+1])
        #output_mask = torch.ne(prediction_all, self.pad_idx)[:, :t+1]  # [batch, t]
        #output_mask = output_mask.type(torch.FloatTensor).to(src.device)

        unfinished_mask_all = torch.cat(unfinished_mask_all, dim=1).type_as(log_selected_token_dist)
        assert unfinished_mask_all.size() == log_selected_token_dist.size()
        #assert output_mask.size() == log_selected_token_dist.size()

        #pred_idx_all = torch.cat(pred_idx_all, dim=1).type(torch.LongTensor).to(src.device)
        #assert pred_idx_all.size() == log_selected_token_dist.size()

        eos_idx_mask_all = torch.cat(eos_idx_mask_all, dim=1).to(src.device)
        assert eos_idx_mask_all.size() == log_selected_token_dist.size()

        #return sample_list, log_selected_token_dist, unfinished_mask_all, pred_idx_all
        """
        if entropy_regularize:
            return sample_list, log_selected_token_dist, unfinished_mask_all, eos_idx_mask_all, entropy
        else:
            return sample_list, log_selected_token_dist, unfinished_mask_all, eos_idx_mask_all
        """
        return sample_list, log_selected_token_dist, unfinished_mask_all, eos_idx_mask_all, entropy, location_of_eos_for_each_batch