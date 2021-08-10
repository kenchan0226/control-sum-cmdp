import torch
from utils.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics, RewardStatistics
import time
from utils.time_log import time_since
from utils.reward import sample_list_to_str_list, compute_batch_reward
from utils import io
import nltk
from cytoolz import concat
from torch.nn import CrossEntropyLoss
from model.seq2seq_exact_length_input import Seq2SeqModelExactLenInput
from model.diversity_attn_seq2seq import Seq2SeqDiversityAttnModel


def evaluate_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            #src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_sent_2d_list, trg, trg_oov, trg_lens, trg_mask, _ = batch
            src = batch['src_tensor']
            src_lens = batch['src_lens']
            src_mask = batch['src_mask']
            src_oov = batch['src_oov_tensor']
            oov_lists = batch['oov_lists']
            src_str_list = batch['src_list_tokenized']
            trg_sent_2d_list = batch['trg_sent_2d_list']
            trg = batch['trg_tensor']
            trg_oov = batch['trg_oov_tensor']
            trg_lens = batch['trg_lens']
            trg_mask = batch['trg_mask']
            position_ids = batch['position_ids']
            if opt.multi_style:
                style_label = batch['style_tensor']  # [batch_size]
                style_label = style_label.to(opt.device)

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)

            start_time = time.time()
            if opt.multi_style:
                decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _ = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, style_label)
            elif isinstance(model, Seq2SeqModelExactLenInput):
                decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _ = model(src, src_lens, trg,
                                                                                               src_oov, max_num_oov,
                                                                                               src_mask, batch['exact_lens'])
            elif isinstance(model, Seq2SeqDiversityAttnModel):
                query_tensor = batch['query_tensor'].to(opt.device)
                query_mask = batch['query_mask'].to(opt.device)
                query_lens = batch['query_lens']
                decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _ = model(
                    src, src_lens, trg, query_tensor, query_lens, src_oov, max_num_oov, src_mask, query_mask)
            else:
                decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _ = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, position_ids)
            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            if opt.copy_attention:  # Compute the loss using target with oov words
                loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, coverage_loss=False)
            else:  # Compute the loss using target without oov words
                loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, coverage_loss=False)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total, loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def evaluate_reward(data_loader, generator, reward_obj, opt):
    """Return the avg. reward in the validation dataset"""
    generator.model.eval()
    final_reward_sum = 0.0
    n_batch = 0
    sample_time_total = 0.0
    reward_type = opt.reward_type

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            # load one2many dataset
            #src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_sent_2d_list, trg, trg_oov, trg_lens, trg_mask, _ = batch
            src = batch['src_tensor']
            src_lens = batch['src_lens']
            src_mask = batch['src_mask']
            src_oov = batch['src_oov_tensor']
            oov_lists = batch['oov_lists']
            src_str_list = batch['src_list_tokenized']
            trg_sent_2d_list = batch['trg_sent_2d_list']
            trg = batch['trg_tensor']
            trg_oov = batch['trg_oov_tensor']
            trg_lens = batch['trg_lens']
            trg_mask = batch['trg_mask']

            control_variables = {}
            if 1 in opt.control_modes:
                control_variables['len_bins'] = batch['len_bins']
            if 7 in opt.control_modes:
                control_variables['reference_entities_list'] = batch['reference_entities_list']
            if 5 in opt.control_modes:
                control_variables['src_word_2d_list'] = batch['src_list_tokenized']
                control_variables['abs_bins'] = batch['abs_bins']

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)

            start_time = time.time()
            # sample a sequence
            # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, preidiction is a list of 0 dim tensors
            if isinstance(generator.model, Seq2SeqModelExactLenInput):
                sample_list, log_selected_token_dist, output_mask, _, _, _ = generator.sample_with_exact_len_input(
                    src, src_lens, src_oov, src_mask, oov_lists, batch['exact_lens'], greedy=True, entropy_regularize=False)
            else:
                sample_list, log_selected_token_dist, output_mask, _, _, _ = generator.sample(
                    src, src_lens, src_oov, src_mask, oov_lists, greedy=True, entropy_regularize=False)

            pred_str_list = sample_list_to_str_list(sample_list, oov_lists, opt.idx2word, opt.vocab_size, io.EOS,
                                                        io.UNK, opt.replace_unk,
                                                        src_str_list)
            sample_time = time_since(start_time)
            sample_time_total += sample_time

            pred_sent_2d_list = []  # each item is a list of predicted sentences (tokenized) for an input sample, used to compute summary level Rouge-l
            trg_sent_2d_list_tokenized = []  # each item is a list of target sentences (tokenized) for an input sample
            trg_str_list = []  # each item is the target output sequence (tokenized) for an input sample
            for pred_str, trg_sent_list in zip(pred_str_list, trg_sent_2d_list):
                pred_sent_list = nltk.tokenize.sent_tokenize(' '.join(pred_str))
                pred_sent_list = [pred_sent.strip().split(' ') for pred_sent in pred_sent_list]
                pred_sent_2d_list.append(pred_sent_list)

                trg_sent_list = [trg_sent.strip().split(' ') for trg_sent in trg_sent_list]
                trg_sent_2d_list_tokenized.append(trg_sent_list)
                trg_str_list.append(list(concat(trg_sent_list)))

            trg_sent_2d_list = trg_sent_2d_list_tokenized

            final_reward = compute_batch_reward(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size,
                                                reward_obj, regularization_factor=0.0, regularization_type=0, entropy=None
                                                , control_variables=control_variables)  # np.array, [batch_size]

            final_reward_sum += final_reward.detach().sum(0).item()

    eval_reward_stat = RewardStatistics(final_reward_sum, pg_loss=0, n_batch=n_batch, sample_time=sample_time_total)

    return eval_reward_stat


