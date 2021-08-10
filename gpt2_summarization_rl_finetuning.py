import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from os.path import join
import json
from cytoolz import curry
from utils.reward import *
from utils.statistics import RewardStatistics, LagrangianStatistics
import time
from utils.time_log import time_since
#from utils.transformers_io import SummarizationDataset
from utils.io import remove_old_epoch_states, truncate_and_tokenize_sent_list, find_latest_epoch_state, LEN_BINS, ext_frag_density_to_bin, n_gram_novelty_to_bin, fusion_ratio_to_bin
from gpt2_summarization_finetuning import SummarizationDataset, get_control_mode_special_ids_dict
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils.masked_softmax import MaskedSoftmax
import torch.nn.functional as F
import nltk
from model.lagrangian import Lagrangian
from rl_pipeline import build_cost_objects, build_reward_object, train_lagrangian_multiplier
from utils.cost import compute_batch_cost
from utils.reward import compute_batch_reward
from utils.report import export_train_and_valid_reward, export_lagrangian_stats
import sys

EPS = 1e-8

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                                  CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'camembert': (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)
}


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = SummarizationDataset(tokenizer, args, data_dir=args.data_dir, split='val' if evaluate else 'train', control_modes=args.control_modes)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


"""
def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)
"""


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


@curry
def coll_rl(batch, TLDR_id_list, pad_idx=0, tokenizer=None, control_modes=[], control_mode_special_ids_dict={}, input_trunc_len=512, output_trunc_len=100):
    #doc_seq_lens = []
    #for doc, _, _, _, _ in batch:
    #    doc_seq_lens.append(len(doc[:input_trunc_len]))
    #max_doc_len = max(doc_seq_lens)

    input_lens = []
    doc_trunc_ids_all = []
    prefix_ids_all = []
    doc_word_2d_list = []
    summary_sent_2d_list_tokenized = []
    doc_sent_2d_list_tokenized = []
    summary_word_2d_list = []
    len_bin_list = []
    abs_bin_list = []
    summary_lens = []
    masked_questions_ids_2dlist = []
    answer_2dlist = []
    #answer_id_2dlist = []
    #multiple_choices_ids_2dlist = []
    reference_entities_list = []

    #max_number_length_token_ids = 3

    for doc, summary, doc_sent_list, summary_sent_list, controllable_fields in batch:
        # handle sentence list
        summary_sent_list_tokenized = truncate_and_tokenize_sent_list(summary_sent_list, output_trunc_len)
        summary_sent_2d_list_tokenized.append(summary_sent_list_tokenized)
        doc_sent_list_tokenized = truncate_and_tokenize_sent_list(doc_sent_list, input_trunc_len)
        doc_sent_2d_list_tokenized.append(doc_sent_list_tokenized)

        # handle doc word list and trg word list
        doc_str = ' '.join(doc_sent_list)
        doc_word_list = doc_str.split(' ')[:input_trunc_len]
        doc_word_2d_list.append(doc_word_list)
        summary_str = ' '.join(summary_sent_list)
        summary_word_list = summary_str.split(' ')[:output_trunc_len]
        summary_word_2d_list.append(summary_word_list)
        summary_len = len(summary_word_list)

        # handle tensors
        doc_truncated = doc[:input_trunc_len]
        # control mode specific operations
        special_token_id_list = []

        for control_mode in control_modes:
            if control_mode == 1:
                len_bin = LEN_BINS[summary_len]
                special_token_id_list.append(control_mode_special_ids_dict['len_bin'][len_bin])
                #print(summary_len)
                #print(len_bin)
                #print(control_mode_special_ids_dict['len_bin'][len_bin])
                len_bin_list.append(len_bin)
            elif control_mode == 2:
                length_token_ids = tokenizer.convert_tokens_to_ids([str(summary_len)])
                special_token_id_list += length_token_ids
                summary_lens.append(summary_len)
            elif control_mode == 5:
                abs_bin = ext_frag_density_to_bin(controllable_fields['ext_frag_density'])
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
                abs_bin_list.append(abs_bin)
            elif control_mode == 4:
                abs_bin = n_gram_novelty_to_bin(controllable_fields['two_gram_novelty'])
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
                abs_bin_list.append(abs_bin)
            elif control_mode == 6:
                abs_bin = fusion_ratio_to_bin(controllable_fields['avg_fusion_ratio'])
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
                abs_bin_list.append(abs_bin)
            elif control_mode == 7:
                #special_token_id_list += controllable_fields['reference_entities_prefix_ids']
                doc_truncated = controllable_fields['reference_entities_prefix_ids'] + doc_truncated
                masked_questions_ids_2dlist.append(controllable_fields['masked_question_ids'])
                answer_2dlist.append(controllable_fields['answer_list'])
                #answer_id_2dlist.append(controllable_fields['answer_idx_list'])
                #multiple_choices_ids_2dlist.append(controllable_fields['multiple_choices_ids'])
                reference_entities_list.append(controllable_fields['reference_entities'])

        doc_trunc_ids_all.append(doc_truncated)
        prefix_ids = TLDR_id_list[:-1] + special_token_id_list + TLDR_id_list[-1:]
        prefix_ids_all.append(prefix_ids)
        input_lens.append( len(doc_truncated) + len(prefix_ids) )

    max_input_len = max(input_lens)
    input_ids_all_padded = []
    position_ids_all_padded = []
    prompt_ids_list = []
    summary_ids_list = []
    for doc_trunc_ids, prefix_ids in zip(doc_trunc_ids_all, prefix_ids_all):
        doc_len = len(doc_trunc_ids)
        padding_length = max_input_len - doc_len - len(prefix_ids)
        input_ids = doc_trunc_ids + ([pad_idx] * padding_length) + prefix_ids
        position_ids = list(range(doc_len + padding_length)) + list(range(doc_len, doc_len + len(prefix_ids)))
        input_ids_all_padded.append(input_ids)
        position_ids_all_padded.append(position_ids)
        #prompt_ids = doc_truncated + TLDR_id_list[:-1] + special_token_id_list + TLDR_id_list[-1:]
        #prompt_ids_list.append(prompt_ids)
        #summary_truncated = summary[:output_trunc_len]
        #summary_ids_list.append(summary_truncated)
    """
    print()
    print(tokenizer.decode(input_ids_all_padded[0], clean_up_tokenization_spaces=False))
    print(tokenizer.decode(input_ids_all_padded[1], clean_up_tokenization_spaces=False))
    print(tokenizer.decode(input_ids_all_padded[2], clean_up_tokenization_spaces=False))
    print(tokenizer.decode(input_ids_all_padded[3], clean_up_tokenization_spaces=False))
    print(input_ids_all_padded[0])
    print(input_ids_all_padded[1])
    print(input_ids_all_padded[2])
    print(input_ids_all_padded[3])
    print()
    print(position_ids_all_padded[0])
    print(position_ids_all_padded[1])
    print(position_ids_all_padded[2])
    print(position_ids_all_padded[3])
    #print()
    #print(reference_entities_list)
    exit()
    """
    input_ids_tensor = torch.LongTensor(input_ids_all_padded)
    position_ids_tensor = torch.LongTensor(position_ids_all_padded)

    #batch = {'input_ids': input_ids_tensor, 'position_ids': position_ids_tensor, 'doc_list_tokenized': document_list_tokenized, 'summary_sent_2d_list': summary_sent_2d_list}
    batch = {'input_ids': input_ids_tensor, 'position_ids': position_ids_tensor}
    batch['prompt_ids_list'] = prompt_ids_list
    batch['doc_sent_2d_list_tokenized'] = doc_sent_2d_list_tokenized
    batch['summary_sent_2d_list_tokenized'] = summary_sent_2d_list_tokenized
    batch['doc_word_2d_list'] = doc_word_2d_list
    batch['summary_word_2d_list'] = summary_word_2d_list
    batch['len_bins'] = len_bin_list
    batch['abs_bins'] = abs_bin_list
    batch['summary_lens'] = summary_lens
    batch['summary_ids_list'] = summary_ids_list
    batch['reference_entities_list'] = reference_entities_list
    batch['masked_questions_ids_2dlist'] = masked_questions_ids_2dlist
    batch['answer_2dlist'] = answer_2dlist
    #batch['answer_id_2dlist'] = answer_id_2dlist
    #batch['multiple_choices_ids_2dlist'] = multiple_choices_ids_2dlist
    return batch

def sample_sequence(model, tokenizer, input_ids, mask, position_ids, max_output_length, eos_idx, is_greedy=False):
    batch_size = input_ids.size(0)
    past = None
    end_flags = [False] * batch_size
    log_selected_token_dist = []
    #device = input_ids.device()
    #print("is greedy: {}".format(is_greedy))

    for t in range(max_output_length):
        #print()
        #print("t: {}".format(t))
        #start_time = time.time()
        inputs = {'input_ids': input_ids, 'past': past, 'attention_mask': mask, 'position_ids': position_ids}
        outputs = model(**inputs)
        #print("input_ids size: {}".format(input_ids.size()))
        #print("attention_mask size: {}".format(mask.size()))
        #print("position_ids size: {}".format(position_ids.size()))
        #print("forward time: {}".format(time_since(start_time)))
        prediction_scores = outputs[0]  # (batch_size, sequence_length, config.vocab_size)
        #print("logit size: {}".format(prediction_scores.size()))
        next_token_logits = prediction_scores[:, -1, :]  # [batch, vocab_size]
        past = outputs[1]  # a list of torch.FloatTensor
        #print("past_size: {}".format(past[0].size(3)))
        #print("prediction_score_size: {}".format(prediction_scores.size()))
        #with torch.no_grad():
        #    next_token_distribution = F.softmax(next_token_logits, dim=-1)  # [batch, vocab_size]
        #next_token_log_distribution = torch.log(next_token_distribution + EPS)  # [batch, vocab_size]
        #start_time = time.time()
        next_token_log_distribution = F.log_softmax(next_token_logits, dim=-1)  # [batch, vocab_size]
        with torch.no_grad():
            next_token_distribution = torch.exp(next_token_log_distribution)
        #print("softmax time: {}".format(time_since(start_time)))
        #start_time = time.time()
        if is_greedy:
            next_token = torch.argmax(next_token_distribution, dim=-1).unsqueeze(-1)  # [batch, 1]
        else:
            next_token = torch.multinomial(next_token_distribution, num_samples=1)  # [batch, 1]
            log_selected_token_dist.append(next_token_log_distribution.gather(1, next_token))
        #print("gather time: {}".format(time_since(start_time)))
        if t == 0:
            generated = next_token
            #log_distributions_all = next_token_log_distribution.unsqueeze(1)  # [batch, 1, vocab_size]
        else:
            generated = torch.cat((generated, next_token), dim=1)
            #log_distributions_all = torch.cat((log_distributions_all, next_token_log_distribution.unsqueeze(1)), dim=1)  # [batch, seq_len, vocab_size]

        for i in range(batch_size):
            if next_token[i, 0].item() == eos_idx:
                end_flags[i] = True
        if all(end_flags):
            break

        input_ids = next_token  # [batch, 1]
        next_position_id = position_ids[:, -1] + 1  # [batch_size]
        position_ids = next_position_id.unsqueeze(1)  # [batch, 1]
        # position_ids = torch.cat([position_ids, next_position_id.unsqueeze(1)], dim=1)
        next_attention_mask = torch.FloatTensor([1] * batch_size).to(mask.device)  # [batch_size]
        mask = torch.cat([mask, next_attention_mask.unsqueeze(1)], dim=1)

    if not is_greedy:
        log_selected_token_dist = torch.cat(log_selected_token_dist, dim=1)  # [batch, T]
        assert log_selected_token_dist.size() == torch.Size([batch_size, t+1])

    #start_time = time.time()
    outputs = generated.tolist()
    output_str_list = []
    output_ids_list = []
    unfinished_mask = torch.ones_like(generated).float()
    for i, out_ids in enumerate(outputs):
        eos_positions = [position for position, word_id in enumerate(out_ids) if word_id == eos_idx]
        if len(eos_positions) > 0:
            end_position = eos_positions[0]
            if end_position < len(out_ids) - 1:
                unfinished_mask[i, end_position+1:] = 0.0
            out_ids = out_ids[:end_position]
        output_ids_list.append(out_ids)
        out_text = tokenizer.decode(out_ids, clean_up_tokenization_spaces=False)
        output_str_list.append(out_text)
    #print("decode time: {}".format(time_since(start_time)))
    #exit()

    return output_str_list, log_selected_token_dist, unfinished_mask, output_ids_list


def tokenize_and_sentence_tokenize_str_list(out_str_list):
    pred_word_2d_list = []
    pred_word_2d_list_sent_tokenized = []
    for output_str in out_str_list:
        pred_word_2d_list.append(output_str.split(' '))
        output_sent_list = nltk.tokenize.sent_tokenize(output_str)
        output_sent_list = [output_sent.strip().split(' ') for output_sent in output_sent_list]
        pred_word_2d_list_sent_tokenized.append(output_sent_list)
    return pred_word_2d_list, pred_word_2d_list_sent_tokenized


def compute_ml_loss(model, prompt_ids_list, target_ids_list, pad_idx, eos_idx, device):
    ml_input_ids_list = []
    ml_label_ids_list = []
    ml_input_ids_lens = []
    for prompt_ids, target_ids in zip(prompt_ids_list, target_ids_list):
        ml_input_ids = prompt_ids + target_ids + [eos_idx]
        ml_input_ids_list.append(ml_input_ids)
        ml_label_ids = [-100] * len(prompt_ids) + target_ids + [eos_idx]
        ml_label_ids_list.append(ml_label_ids)
        ml_input_ids_lens.append(len(ml_input_ids))
    # padding
    max_ml_input_ids_len = max(ml_input_ids_lens)
    ml_input_ids_list_padded = []
    ml_label_ids_list_padded = []
    for ml_input_ids, ml_label_ids in zip(ml_input_ids_list, ml_label_ids_list):
        ml_padding_length = max_ml_input_ids_len - len(ml_input_ids)
        ml_input_ids_list_padded.append(ml_input_ids + [pad_idx] * ml_padding_length)
        ml_label_ids_list_padded.append(ml_label_ids + [-100] * ml_padding_length)
    #print("input_ids sizes: ")
    #[print(len(ml_input_ids))for ml_input_ids in ml_input_ids_list_padded]
    #[print(len(ml_label_ids)) for ml_label_ids in ml_label_ids_list_padded]
    #[print(ml_input_ids) for ml_input_ids in ml_input_ids_list_padded]
    #[print(ml_label_ids) for ml_label_ids in ml_label_ids_list_padded]
    #print("target_ids sizes: ")
    #[print(len(ml_target_ids)) for ml_target_ids in target_ids_list]
    #exit()
    ml_input_ids_tensor = torch.LongTensor(ml_input_ids_list_padded).to(device)
    ml_label_ids_tensor = torch.LongTensor(ml_label_ids_list_padded).to(device)
    ml_attn_mask = torch.ne(ml_input_ids_tensor, pad_idx).float()
    ml_outputs = model(input_ids=ml_input_ids_tensor, attention_mask=ml_attn_mask, labels=ml_label_ids_tensor)
    ml_loss = ml_outputs[0]
    return ml_loss


def train_rl(args, train_dataset, model, tokenizer, lagrangian_model, epoch_state_dict, control_mode_special_ids_dict):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
        if args.continue_training:
            report_train_reward_statistics = epoch_state_dict['report_train_reward_statistics']
            report_train_reward = epoch_state_dict['report_train_reward']
            report_valid_reward = epoch_state_dict['report_valid_reward']
            best_valid_reward = epoch_state_dict['best_valid_reward']
            previous_valid_reward = epoch_state_dict['previous_valid_reward']
            num_stop_increasing = epoch_state_dict['num_stop_increasing']
            print("Previous valid rewards: {}".format(report_valid_reward))
        else:
            report_train_reward_statistics = RewardStatistics()
            report_train_reward = []
            report_valid_reward = []
            best_valid_reward = float('-inf')
            previous_valid_reward = float('-inf')
            num_stop_increasing = 0

    if args.continue_training:
        global_step = epoch_state_dict['global_step']
        epoch = epoch_state_dict['epoch']
    else:
        global_step = 0
        epoch = 0

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<control><summarize>:"))
    TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("TL;DR:"))
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    pad_idx = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
    eos_idx = tokenizer.eos_token_id
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=coll_rl(TLDR_id_list=TLDR_id_list, pad_idx=pad_idx, tokenizer=tokenizer, control_modes=args.control_modes,
                                                     control_mode_special_ids_dict=control_mode_special_ids_dict,
                                                     input_trunc_len=args.input_trunc_length, output_trunc_len=args.output_trunc_length))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                            num_training_steps=t_total)

    if args.continue_training:
        optimizer.load_state_dict(epoch_state_dict['optimizer'])
        #if not args.do_not_reload_scheduler:
        #    scheduler.load_state_dict(epoch_state_dict['scheduler'])

    reward_obj = build_reward_object(args.reward_type, args.device)

    if args.constrained_mdp:
        optimizer_lagrangian = torch.optim.Adam(params=filter(lambda p: p.requires_grad, lagrangian_model.parameters()), lr=args.learning_rate_multiplier)
        pretrained_model_args = {}
        pretrained_model_args['pad_idx'] = pad_idx
        pretrained_model_args['eos_idx'] = eos_idx
        pretrained_model_args['TLDR_ids_list'] = TLDR_id_list
        cost_objs = build_cost_objects(args.cost_types, args.device, args.train_batch_size, args.cost_thresholds, model, pretrained_model_args)
        if args.continue_training:
            optimizer_lagrangian.load_state_dict(epoch_state_dict['optimizer_lagrangian'])
            report_train_lagrangian_statistics = epoch_state_dict['report_train_lagrangian_statistics']
            report_lagrangian_loss = epoch_state_dict['report_lagrangian_loss']
            report_lagrangian_multipliers = epoch_state_dict['report_lagrangian_multipliers']
            report_violate_amounts = epoch_state_dict['report_violate_amounts']
            report_lagrangian_grad_norms = epoch_state_dict['report_lagrangian_grad_norms']
        else:
            report_train_lagrangian_statistics = LagrangianStatistics()
            report_lagrangian_loss = []
            report_lagrangian_multipliers = []
            report_violate_amounts = []
            report_lagrangian_grad_norms = []

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        if args.constrained_mdp:
            lagrangian_model = torch.nn.DataParallel(lagrangian_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True,
                                                          broadcast_buffers=False)
        if args.constrained_mdp:
            lagrangian_model = torch.nn.parallel.DistributedDataParallel(lagrangian_model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True,
                                                                         broadcast_buffers=False)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("logging_steps = %d", args.logging_steps)


    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            context = batch['input_ids']
            position_ids = batch['position_ids']
            src_sent_2d_list_tokenized = batch['doc_sent_2d_list_tokenized']
            trg_sent_2d_list_tokenized = batch['summary_sent_2d_list_tokenized']
            src_word_2d_list = batch['doc_word_2d_list']
            trg_word_2d_list = batch['summary_word_2d_list']
            prompt_ids_list = batch['prompt_ids_list']
            trg_summary_ids_list = batch['summary_ids_list']

            mask = torch.ne(context, pad_idx).float()
            batch_size = context.size(0)
            context = context.to(args.device)
            position_ids = position_ids.to(args.device)
            mask = mask.to(args.device)
            input_ids = context

            control_variables = {}
            """
            if 1 in args.control_modes:
                control_variables['len_bins'] = batch['len_bins']
            if 2 in args.control_modes:
                control_variables['exact_lens'] = batch['exact_lens']
            if 3 in args.control_modes or 4 in args.control_modes or 5 in args.control_modes or 6 in args.control_modes:
                control_variables['abs_bins'] = batch['abs_bins']
                if 6 in args.control_modes:
                    # tokenize the each src sentence list in the batch and put it in the control variable.
                    control_variables['src_word_2d_list_sent_tokenized'] = src_sent_2d_list_tokenized
                # control_variables['src_word_2d_list'] = batch['src_list_tokenized']
            # if 10 in opt.cost_types or 11 in opt.cost_types or 18 in opt.cost_types or 19 in opt.cost_types:
            """
            for control_mode in args.control_modes:
                if control_mode == 1:
                    control_variables['len_bins'] = batch['len_bins']
                elif control_mode == 2:
                    control_variables['exact_lens'] = batch['summary_lens']
                elif control_mode == 5 or control_mode == 4 or control_mode == 6:
                    control_variables['abs_bins'] = batch['abs_bins']
                elif control_mode == 7:
                    control_variables['reference_entities_list'] = batch['reference_entities_list']
                    control_variables['masked_questions_ids_2dlist'] = batch['masked_questions_ids_2dlist']
                    control_variables['answer_2dlist'] = batch['answer_2dlist']
                    #control_variables['answer_id_2dlist'] = batch['answer_id_2dlist']
                    #control_variables['multiple_choices_ids_2dlist'] = batch['multiple_choices_ids_2dlist']

            control_variables['src_word_2d_list'] = src_word_2d_list
            control_variables['src_word_2d_list_sent_tokenized'] = src_sent_2d_list_tokenized

            # sample sequence
            model.train()
            start_time = time.time()
            sample_output_str_list, log_distributions_all, output_mask, sample_output_ids_list = \
                sample_sequence(model, tokenizer, input_ids, mask, position_ids, args.max_output_length, eos_idx, is_greedy=False)
            #input_sequence_len = context.size(1)
            #print("input len {}".format(input_sequence_len))
            #outputs_tensor = model.generate(input_ids=input_ids, max_length=input_sequence_len+output_trunc_len, do_sample=True, num_beams=1, pad_token_id=pad_idx, eos_token_ids=[eos_idx], position_ids=position_ids, attn_mask=mask)
            sample_time = time_since(start_time)
            #print("sample_time: {}".format(sample_time))

            # log_distributions_all: [batch, max_seq_len], output_mask: [batch, max_seq_len]
            # sum, apply masking and pass it to the control variables
            # prompt ids
            log_distributions_all_sum = (log_distributions_all * output_mask).sum(dim=1)  # [batch]
            #print("log distribution size: ")
            #print(log_distributions_all.size())
            control_variables['pred_log_probs_sum'] = log_distributions_all_sum
            control_variables['pred_ids_list'] = sample_output_ids_list
            control_variables['prompt_ids_list'] = prompt_ids_list

            # greedy sequence
            start_time = time.time()
            with torch.no_grad():
                model.eval()
                greedy_output_str_list, _, _, _ = \
                    sample_sequence(model, tokenizer, input_ids, mask, position_ids, args.max_output_length, eos_idx, is_greedy=True)
            #print("greedy_time: {}".format(time_since(start_time)))

            sample_word_2d_list, sample_sent_2d_list_tokenized = tokenize_and_sentence_tokenize_str_list(sample_output_str_list)
            greedy_word_2d_list, greedy_sent_2d_list_tokenized = tokenize_and_sentence_tokenize_str_list(greedy_output_str_list)
            max_sample_seq_len = log_distributions_all.size(1)

            start_time = time.time()
            with torch.no_grad():
                cumulative_reward = compute_batch_reward(sample_word_2d_list, sample_sent_2d_list_tokenized, trg_word_2d_list,
                                                         trg_sent_2d_list_tokenized, batch_size, reward_obj,
                                                         control_variables=control_variables)
                # store the sum of cumulative reward (before baseline) for the experiment log
                cumulative_reward_sum = cumulative_reward.detach().sum(0).item()

                baseline = compute_batch_reward(greedy_word_2d_list, greedy_sent_2d_list_tokenized, trg_word_2d_list,
                                                         trg_sent_2d_list_tokenized, batch_size, reward_obj,
                                                         control_variables=control_variables)
                if args.constrained_mdp:
                    cumulative_cost = compute_batch_cost(sample_word_2d_list, sample_sent_2d_list_tokenized, trg_word_2d_list,
                                                         trg_sent_2d_list_tokenized, batch_size, cost_objs,
                                                         control_variables)  # [sample_batch_size, num_cost_types]
                    cumulative_cost_mean = cumulative_cost.mean(0)  # [num_cost_types]

                    # cumulative_cost: [sample_batch_size, len(cost_types)]
                    # subtract the regularization term: \lambda \dot C_t
                    lagrangian_model_to_compute = lagrangian_model.module if hasattr(lagrangian_model, 'module') else lagrangian_model
                    constraint_regularization = lagrangian_model_to_compute.compute_regularization(
                        cumulative_cost)  # [sample_batch_size]
                    cumulative_reward -= constraint_regularization
                    #exit()
                    #print(cumulative_cost.detach().cpu().numpy())
                    #print(constraint_regularization.detach().cpu().numpy())

                # Subtract the cumulative reward by a baseline if needed
                if args.baseline != 'none':
                    cumulative_reward = cumulative_reward - baseline  # [sample_batch_size]
                # q value estimation for each time step equals to the (baselined) cumulative reward
                q_value_estimate = cumulative_reward.unsqueeze(1).repeat(1, max_sample_seq_len)  # [sample_batch_size, max_pred_seq_len]

            #print(log_distributions_all.size())
            #print(q_value_estimate.size())
            #print(cumulative_reward.detach().cpu().numpy())
            #print(baseline.detach().cpu().numpy())
            #print(cumulative_reward_sum)

            # compute loss for model
            q_estimate_compute_time = time_since(start_time)
            #print("q_estimate_compute_time: {}".format(q_estimate_compute_time))
            q_value_estimate.requires_grad_(True)

            # compute the policy gradient objective
            #print(log_distributions_all.size())
            #print(output_mask.size())
            #print(q_value_estimate.size())
            start_time = time.time()
            pg_loss = compute_pg_loss(log_distributions_all, output_mask, q_value_estimate)

            pg_loss_normalized = pg_loss.div(batch_size)
            #print("pg_loss compute time: {}".format(time_since(start_time)))
            start_time = time.time()

            #if args.gradient_accumulation_steps > 1:
            #    pg_loss_normalized = pg_loss_normalized / args.gradient_accumulation_steps

            if args.ml_loss_coefficient > 0:
                model.train()
                ml_loss = compute_ml_loss(model, prompt_ids_list, trg_summary_ids_list, pad_idx, eos_idx, args.device)
                if args.n_gpu > 1:
                    ml_loss = ml_loss.mean()  # mean() to average on multi-gpu parallel training
                #if args.gradient_accumulation_steps > 1:
                #    ml_loss = ml_loss / args.gradient_accumulation_steps
                total_loss = (1 - args.ml_loss_coefficient) * pg_loss_normalized + args.ml_loss_coefficient * ml_loss
            else:
                total_loss = pg_loss_normalized

            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            backward_time = time_since(start_time)
            #print("backward time: {}".format(backward_time))
            #exit()
            if args.local_rank in [-1, 0]:
                batch_reward_stat = RewardStatistics(cumulative_reward_sum, pg_loss.item(), batch_size, sample_time,
                                                     q_estimate_compute_time, backward_time)

                report_train_reward_statistics.update(batch_reward_stat)

            tr_loss += total_loss.item()

            # compute loss for lagrangian model
            if args.constrained_mdp:
                lagrangian_loss, violate_amount = lagrangian_model(cumulative_cost)
                if args.n_gpu > 1:
                    lagrangian_loss = lagrangian_loss.mean()
                    violate_amount = violate_amount.sum()
                lagrangian_loss_normalized = lagrangian_loss.div(batch_size)
                if args.gradient_accumulation_steps > 1:
                    lagrangian_loss_normalized = lagrangian_loss_normalized / args.gradient_accumulation_steps
                lagrangian_loss_normalized.backward()
                if args.local_rank in [-1, 0]:
                    lagrangian_grad_norm = lagrangian_model_to_compute.lagrangian_multiplier.grad.detach().sum().item()
                    batch_lagrangian_stat = LagrangianStatistics(lagrangian_loss=lagrangian_loss.item(), n_batch=batch_size,
                                                           lagrangian_grad_norm=lagrangian_grad_norm,
                                                           violate_amount=violate_amount.item())
                    report_train_lagrangian_statistics.update(batch_lagrangian_stat)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # take a gradient step on model
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    #grad_norm_before_clipping = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                #scheduler.step()
                model.zero_grad()

                # take a gradient step on model on lagrangian model
                if args.constrained_mdp:
                    optimizer_lagrangian.step()
                    lagrangian_model_to_compute.clamp_lagrangian_multiplier()
                    lagrangian_model.zero_grad()

                # increment global step
                global_step += 1

                #if global_step == 10:
                #    stat = torch.cuda.memory_stats("cuda:0")
                #    print()
                #    print()
                #    print(stat['reserved_bytes.all.peak'])
                #    print(stat['allocated_bytes.all.peak'])
                #    exit()

                # log each loss to tensorboard
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # tensorboard
                    #if args.ml_loss_coefficient > 0:
                    #    tb_writer.add_scalar('ml_loss', ml_loss.item(), global_step)
                    #tb_writer.add_scalar('pg_loss', pg_loss_normalized.item(), global_step)
                    #tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', tr_loss - logging_loss, global_step)
                    logging_loss = tr_loss

                    if args.constrained_mdp:
                        lambda_tensor = lagrangian_model_to_compute.get_lagrangian_multiplier()
                        for cost_i, cost_type in enumerate(args.cost_types):
                            tb_writer.add_scalar('cost_{}'.format(cost_type),
                                                 cumulative_cost_mean[cost_i].detach().item(), global_step)
                            tb_writer.add_scalar('lambda_{}'.format(cost_type), lambda_tensor[cost_i].item(),
                                                 global_step)
                    # statistics objects
                    current_train_reward = report_train_reward_statistics.reward()
                    current_train_pg_loss = report_train_reward_statistics.loss()
                    report_train_reward.append(current_train_reward)
                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, step, global_step))
                    logging.info(
                       'avg training reward: %.5f; avg training loss: %.5f' % (
                           current_train_reward, current_train_pg_loss))
                    if args.constrained_mdp:
                        current_lagrangian_loss = report_train_lagrangian_statistics.loss()
                        current_lagrangian_grad_norm = report_train_lagrangian_statistics.grad_norm()
                        current_violate_amount = report_train_lagrangian_statistics.violate_amt()
                        report_lagrangian_loss.append(current_lagrangian_loss)
                        report_violate_amounts.append(current_violate_amount)
                        report_lagrangian_grad_norms.append(current_lagrangian_grad_norm)
                        lagrangian_multipliers_array = lagrangian_model_to_compute.get_lagrangian_multiplier_array()
                        report_lagrangian_multipliers.append(lagrangian_multipliers_array)
                        logging.info("Lagrangian_loss: %.5f; grad_norm: %.5f" % (
                        current_lagrangian_loss, current_lagrangian_grad_norm))
                        logging.info("Value of lagrangian_multipliers: {}".format(lagrangian_multipliers_array))

                    report_train_reward_statistics.clear()
                    if args.constrained_mdp:
                        report_train_lagrangian_statistics.clear()

                # check point
                """
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    lagrangian_model_to_save = lagrangian_model.module if hasattr(lagrangian_model,
                                                                                  'module') else lagrangian_model
                    if args.constrained_mdp:
                        torch.save(lagrangian_model_to_save.state_dict(), os.path.join(output_dir, "lagrangian_model.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    #torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                """
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            logging.info("Finished epoch {}".format(epoch))

            # save epoch state for continue training
            if epoch % 1 == 0:
                current_epoch_state_dir = os.path.join(args.output_dir, 'epoch_states', '{}-epoch'.format(epoch))
                if not os.path.exists(current_epoch_state_dir):
                    os.makedirs(current_epoch_state_dir)
                # save model
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(current_epoch_state_dir)
                # save epoch state dict
                lagrangian_model_to_save = lagrangian_model.module if hasattr(lagrangian_model,
                                                                              'module') else lagrangian_model
                current_epoch_state = {
                    'epoch': epoch,
                    # 'total_batch': total_batch,
                    # 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict(),
                    'lagrangian_model': lagrangian_model_to_save.state_dict() if args.constrained_mdp else None,
                    'optimizer_lagrangian': optimizer_lagrangian.state_dict() if args.constrained_mdp else None,
                    'best_valid_reward': best_valid_reward,
                    'previous_valid_reward': previous_valid_reward,
                    'num_stop_increasing': num_stop_increasing,
                    'report_train_reward_statistics': report_train_reward_statistics,
                    'report_train_reward': report_train_reward,
                    'report_valid_reward': report_valid_reward,
                    'report_train_lagrangian_statistics': report_train_lagrangian_statistics if args.constrained_mdp else None,
                    'report_lagrangian_loss': report_lagrangian_loss if args.constrained_mdp else None,
                    'report_lagrangian_multipliers': report_lagrangian_multipliers if args.constrained_mdp else None,
                    'report_violate_amounts': report_violate_amounts if args.constrained_mdp else None,
                    'report_lagrangian_grad_norms': report_lagrangian_grad_norms if args.constrained_mdp else None,
                    'global_step': global_step
                }
                epoch_state_dict_path = os.path.join(current_epoch_state_dir, 'epoch_state_dict.pt')
                torch.save(  # save epoch states
                    current_epoch_state,
                    open(epoch_state_dict_path, 'wb')
                )
                logging.info("saved epoch state.")

            # run validation and save validation stat for every epoch
            valid_reward_stat = evaluate(args, model, tokenizer, reward_obj, control_mode_special_ids_dict=control_mode_special_ids_dict)
            current_valid_reward = valid_reward_stat.reward()
            report_valid_reward.append(current_valid_reward)
            # print out valid reward
            logging.info(
                'avg validation reward: %.5f; best validation reward: %.5f' % (
                    current_valid_reward, best_valid_reward))
            lagrangian_multipliers_array = lagrangian_model_to_compute.get_lagrangian_multiplier_array()
            logging.info('Value of lagrangian_multipliers: %s' % (lagrangian_multipliers_array.tostring()))

            if current_valid_reward > previous_valid_reward:  # update the best valid reward and save the model parameters
                logging.info("Valid reward increases")
                if current_valid_reward > best_valid_reward:
                    best_valid_reward = current_valid_reward
                num_stop_increasing = 0
            else:
                logging.info("Valid reward does not increases")
                num_stop_increasing += 1

            previous_valid_reward = current_valid_reward


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        if args.max_epochs > 0 and epoch >= args.max_epochs:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        # export the training curve
        train_valid_curve_path = join(args.output_dir, 'train_valid_curve')
        export_train_and_valid_reward(report_train_reward, report_valid_reward, args.logging_steps,
                                      train_valid_curve_path)
        if args.constrained_mdp:
            #print()
            #print(report_lagrangian_multipliers)
            export_lagrangian_stats(report_lagrangian_loss, report_lagrangian_multipliers, report_lagrangian_grad_norms,
                                    report_violate_amounts, args.logging_steps, train_valid_curve_path)

        # log best reward
        logging.info("final_best_valid_reward: %.3f" % best_valid_reward)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, reward_obj, prefix="", control_mode_special_ids_dict={}):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    #args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    #TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<control><summarize>:"))
    TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("TL;DR:"))
    pad_idx = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
    eos_idx = tokenizer.eos_token_id
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=coll_rl(TLDR_id_list=TLDR_id_list, pad_idx=pad_idx, tokenizer=tokenizer, control_modes=args.control_modes,
                                                     control_mode_special_ids_dict=control_mode_special_ids_dict,
                                                    input_trunc_len=args.input_trunc_length, output_trunc_len=args.output_trunc_length))

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    valid_reward_sum = 0.0
    n_batch = 0
    sample_time_total = 0.0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        context = batch['input_ids']
        position_ids = batch['position_ids']
        src_word_2d_list = batch['doc_word_2d_list']
        trg_word_2d_list = batch['summary_word_2d_list']
        src_sent_2d_list_tokenized = batch['doc_sent_2d_list_tokenized']
        trg_sent_2d_list_tokenized = batch['summary_sent_2d_list_tokenized']

        mask = torch.ne(context, pad_idx).float()
        batch_size = context.size(0)
        # construct label, set all the pad_idx to -1 so that it will not be computed in the loss
        #labels = inputs.masked_fill(inputs == pad_idx, -1)
        context = context.to(args.device)
        position_ids = position_ids.to(args.device)
        mask = mask.to(args.device)
        input_ids = context
        n_batch += batch_size

        control_variables = {}
        control_variables['src_word_2d_list'] = src_word_2d_list

        for control_mode in args.control_modes:
            if control_mode == 1:
                control_variables['len_bins'] = batch['len_bins']
            elif control_mode == 2:
                control_variables['exact_lens'] = batch['summary_lens']
            elif control_mode == 5 or control_mode == 4 or control_mode == 6:
                control_variables['abs_bins'] = batch['abs_bins']
            elif control_mode == 7:
                control_variables['reference_entities_list'] = batch['reference_entities_list']
                control_variables['masked_questions_ids_2dlist'] = batch['masked_questions_ids_2dlist']
                control_variables['answer_2dlist'] = batch['answer_2dlist']
                # control_variables['answer_id_2dlist'] = batch['answer_id_2dlist']
                # control_variables['multiple_choices_ids_2dlist'] = batch['multiple_choices_ids_2dlist']

        # greedy sequence
        start_time = time.time()
        with torch.no_grad():
            greedy_output_str_list, _, _, _ = \
                sample_sequence(model, tokenizer, input_ids, mask, position_ids, args.max_output_length, eos_idx, is_greedy=True)

        sample_time = time_since(start_time)
        sample_time_total += sample_time

        greedy_word_2d_list, greedy_sent_2d_list_tokenized = tokenize_and_sentence_tokenize_str_list(
            greedy_output_str_list)


        with torch.no_grad():
            valid_reward = compute_batch_reward(greedy_word_2d_list, greedy_sent_2d_list_tokenized, trg_word_2d_list,
                                            trg_sent_2d_list_tokenized, batch_size, reward_obj,
                                            control_variables=control_variables)

            valid_reward_sum += valid_reward.detach().sum(0).item()

    eval_reward_stat = RewardStatistics(valid_reward_sum, pg_loss=0, n_batch=n_batch, sample_time=sample_time_total)
    return eval_reward_stat


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    #parser.add_argument("--train_data_file", default=None, type=str, required=True,
    #                    help="The input training data file (a text file).")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The path of the directory containing all the splits.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    #parser.add_argument("--eval_data_file", default=None, type=str,
    #                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--max_epochs", default=-1, type=int,
                        help="If > 0: set the maximum number of epochs to perform. Override num_training_epochs. It is used to for continue training. ")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    #parser.add_argument('--save_steps', type=int, default=50,
    #                    help="Save checkpoint every X updates steps.")
    #parser.add_argument('--save_total_limit', type=int, default=None,
    #                    help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--constrained_mdp', action='store_true',
                        help="Use constrainted mdp")
    parser.add_argument('--cost_types', nargs='+', default=[], type=int,
                        help=""" Specify a list of cost function. 
                            Type of cost function. 0: number of 3-gram repeat. 1: min len penalty. Only effective when using constrained mdp.""")
    parser.add_argument('--cost_thresholds', nargs='+', default=[], type=float,
                        help=""" Specify a list of thresholds. Only effective when using constrained mdp.""")
    parser.add_argument('--lagrangian_init_val', type=float, default=0.0,
                        help="The initial value of the lagrangian multiplier. ")
    parser.add_argument('--use_lagrangian_hinge_loss', action="store_true",
                        help="Use hinge loss in lagrangian. ")
    parser.add_argument('--learning_rate_multiplier', type=float, default=0.00001,
                        help="""Starting learning rate of lagrangian multiplier.""")
    parser.add_argument('--control_modes', nargs='+', default=[], type=int,
                        help='0: nothing. 1: control length. 2: control exact length. 3: novel 3-gram range')
    parser.add_argument('--max_output_length', type=int, default=75,
                        help='Max length of output.')
    parser.add_argument('--input_trunc_length', type=int, default=512,
                        help='Max length of output.')
    parser.add_argument('--output_trunc_length', type=int, default=100,
                        help='Max length of output.')
    parser.add_argument('--reward_type', type=int, default=0,
                        help='reward types.')
    parser.add_argument('--baseline', default="self", choices=["none", "self"],
                        help='The baseline in RL training. none: no baseline; self: use greedy decoding as baseline. ')
    parser.add_argument('--continue_training', action="store_true",
                        help="Train from a saved epoch.")
    parser.add_argument('--num_freeze_layers', type=int, default=0,
                        help='number of layers in GPT2 to freeze.')
    parser.add_argument('--ml_loss_coefficient', type=float, default=0.0,
                        help='coefficient for ml training loss.')
    #parser.add_argument('--do_not_reload_scheduler', action="store_true",
    #                    help='Do not reload the state of optimization scheduler. Only effective when using continue training.')
    #parser.add_argument('--train_from', default="",
    #                    help="Train from a saved epoch.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    #torch.autograd.set_detect_anomaly(True)

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    #if args.eval_data_file is None and args.do_eval:
    #    raise ValueError(
    #        "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
    #        "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename=join('exp', args.output_dir.split("/")[-1] + ".log" ),
                        filemode='w')
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        os.makedirs(join(args.output_dir, 'epoch_states'))
        os.makedirs(join(args.output_dir, 'train_valid_curve'))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    # add pad token to special token
    #special_tokens_dict = {'pad_token': '<pad>'}
    #tokenizer.add_special_tokens(special_tokens_dict)
    #model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    # freeze other layers, only train the LM head

    for param in model.transformer.wte.parameters():
        param.requires_grad = False
    for param in model.transformer.wpe.parameters():
        param.requires_grad = False
    if args.num_freeze_layers > 12:
        args.num_freeze_layers = 12
    freeze_layers = list(range(args.num_freeze_layers))
    for layer_idx in freeze_layers:
        for param in model.transformer.h[layer_idx].parameters():
            param.requires_grad = False
        print("Freezed Layer: ", layer_idx)
    #for param in model.transformer.ln_f.parameters():
    #    param.requires_grad = False

    #for param in model.transformer.parameters():
    #    param.requires_grad = False
    #for name, param in model.transformer.named_parameters():
    #    print(name)
    #exit()

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # get ids of special tokens for different control modes
    control_mode_special_ids_dict = get_control_mode_special_ids_dict(args.control_modes, tokenizer)
    print()
    print(control_mode_special_ids_dict)

    # Load epoch dict
    if args.continue_training:
        epoch_state_dict = torch.load(join(args.model_name_or_path, 'epoch_state_dict.pt'), map_location=lambda storage, loc: storage)
        logger.info("Load state dict in {}".format(args.model_name_or_path))
        if args.constrained_mdp:
            lagrangian_model = Lagrangian(len(args.cost_types), args.cost_thresholds, args.lagrangian_init_val,
                                          args.use_lagrangian_hinge_loss)
            lagrangian_model.load_state_dict(epoch_state_dict['lagrangian_model'])
            lagrangian_model.to(args.device)
        else:
            lagrangian_model = None
    else:
        epoch_state_dict = None
        if args.constrained_mdp:
            lagrangian_model = Lagrangian(len(args.cost_types), args.cost_thresholds, args.lagrangian_init_val,
                                          args.use_lagrangian_hinge_loss)
            lagrangian_model.to(args.device)
        else:
            lagrangian_model = None

    # save tokenizer before training:
    if args.local_rank in [-1, 0]:
        tokenizer.save_pretrained(args.output_dir)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train_rl(args, train_dataset, model, tokenizer, lagrangian_model, epoch_state_dict, control_mode_special_ids_dict)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        #tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        reward_obj = build_reward_object(args.reward_type, args.device)
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, reward_obj, control_mode_special_ids_dict=control_mode_special_ids_dict)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()


