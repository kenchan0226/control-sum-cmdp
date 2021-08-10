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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils.masked_softmax import MaskedSoftmax
import torch.nn.functional as F
import nltk
from utils.io import LEN_BINS, ext_frag_density_to_bin, n_gram_novelty_to_bin, fusion_ratio_to_bin, find_latest_epoch_state, remove_old_epoch_states

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  )


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}


class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data_dir, split, control_modes, is_multi_reference=False):
        split_dir = os.path.join(data_dir, split)
        cached_features_file = os.path.join(data_dir, args.model_type + '_cached_lm_' + split)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", data_dir)

            self.examples = []

            n_data = _count_data(split_dir)

            for i in range(n_data):
                js = json.load(open(join(split_dir, '{}.json'.format(i))))
                if js['article'] and js['abstract']:
                    doc_sent_list = js['article']
                    doc_str = ' '.join(doc_sent_list)
                    #doc_word_list = doc_str.split(' ')
                    #doc_str_truncated  = ' '.join(doc_word_list[:MAX_DOC_LEN])

                    summary_sent_list = js['abstract']
                    if is_multi_reference:
                        summary_sent_list = summary_sent_list[0]
                    summary_str = ' '.join(summary_sent_list)
                    #summary_word_list = summary_str.split(' ')
                    #summary_str_truncated = ' '.join(summary_word_list[:MAX_SUMM_LEN])

                    controllable_fields = {}
                    # abstractiveness
                    if 4 in control_modes or 5 in control_modes or 6 in control_modes:
                        controllable_fields['ext_frag_density'] = js['extractive_fragment_density']
                        controllable_fields['avg_fusion_ratio'] = js['avg_fusion_ratio']
                        controllable_fields['two_gram_novelty'] = js['two_gram_novelty']

                    # entities
                    if 7 in control_modes:
                        controllable_fields['reference_entities'] = js['reference_entity_list_non_numerical']
                        controllable_fields['masked_question_ids'] = js['masked_question_ids_list']
                        #controllable_fields['multiple_choices_ids'] = js['multiple_choices_ids_list']
                        #controllable_fields['answer_idx_list'] = js['answer_idx_list']
                        controllable_fields['answer_list'] = js['answer_list']
                        reference_entity_list_non_numerical_str = js["reference_entity_list_non_numerical_str"]
                        controllable_fields['reference_entities_prefix_ids'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(reference_entity_list_non_numerical_str))

                    #doc_str_truncated = doc_str_truncated + " TL;DR:"
                    doc_str_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc_str))
                    #doc_str_tokenized = doc_str_tokenized[:MAX_DOC_LEN+4]  # account for TL;DR:
                    summary_str_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(summary_str))
                    #summary_str_tokenized = summary_str_tokenized[:MAX_SUMM_LEN]

                    #text = doc_str_truncated + " TL;DR: " + summary_str_truncated
                    #tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                    self.examples.append( (tokenizer.build_inputs_with_special_tokens(doc_str_tokenized), tokenizer.build_inputs_with_special_tokens(summary_str_tokenized), doc_sent_list, summary_sent_list, controllable_fields) )

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = SummarizationDataset(tokenizer, args, data_dir=args.data_dir, split='val' if evaluate else 'train', control_modes=args.control_modes)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

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
def coll_ml(batch, TLDR_id_list, pad_idx, eos_idx, tokenizer, control_modes=[], control_mode_special_ids_dict={}, input_trunc_len=512, output_trunc_len=100):
    #for doc, summary, _, _, _ in batch:
    #    seq_lens.append( len(doc[:input_trunc_len]) + len(summary[:output_trunc_len]) )  # needs to add TDLR and EOS
    #max_len = max(seq_lens)
    #max_len = max_len + 5 + len(control_modes)  # add TL;DR:, eos, and special tokens
    #if 2 in control_modes:  # for exact length control, a number may have three tokens
    #    max_len += 2
    input_ids_all = []
    label_ids_all = []
    input_lens = []
    #len_bin_list = []
    for doc, summary, _, summary_sent_list, controllable_fields in batch:
        # process str
        summary_str = ' '.join(summary_sent_list)
        summary_word_list = summary_str.split(' ')[:output_trunc_len]
        summary_len = len(summary_word_list)
        # process tensors
        doc_truncated = doc[:input_trunc_len]
        summary_truncated = summary[:output_trunc_len]
        # control mode specific operations
        special_token_id_list = []
        entity_id_list = []
        for control_mode in control_modes:
            if control_mode == 1:
                len_bin = LEN_BINS[summary_len]
                special_token_id_list.append(control_mode_special_ids_dict['len_bin'][len_bin])
            elif control_mode == 2:
                length_token_ids = tokenizer.convert_tokens_to_ids([str(summary_len)])
                special_token_id_list += length_token_ids
            elif control_mode == 5:
                abs_bin = ext_frag_density_to_bin(controllable_fields['ext_frag_density'])
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
            elif control_mode == 4:
                abs_bin = n_gram_novelty_to_bin(controllable_fields['two_gram_novelty'])
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
            elif control_mode == 6:
                abs_bin = fusion_ratio_to_bin(controllable_fields['avg_fusion_ratio'])
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
            elif control_mode == 7:
                entity_id_list += controllable_fields['reference_entities_prefix_ids']

        input_ids = entity_id_list + doc_truncated + TLDR_id_list[:-1] + special_token_id_list + TLDR_id_list[-1:] + summary_truncated + [eos_idx]
        label_ids = [-100] * ( len(entity_id_list) + len(doc_truncated) + len(TLDR_id_list) + len(special_token_id_list) ) + summary_truncated + [eos_idx]
        input_ids_all.append(input_ids)
        label_ids_all.append(label_ids)
        input_lens.append(len(input_ids))

    max_input_len = max(input_lens)
    input_ids_all_padded = []
    label_ids_all_padded = []
    for input_ids, label_ids in zip(input_ids_all, label_ids_all):
        padding_length = max_input_len - len(input_ids)
        input_ids = input_ids + ([pad_idx] * padding_length)
        label_ids = label_ids + ([-100] * padding_length)
        input_ids_all_padded.append(input_ids)
        label_ids_all_padded.append(label_ids)

    """
    print()
    print()
    print(tokenizer.decode(input_ids_all_padded[0], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(input_ids_all_padded[1], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(input_ids_all_padded[2], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(input_ids_all_padded[3], clean_up_tokenization_spaces=False))

    print(input_ids_all_padded[0])
    print(input_ids_all_padded[1])
    print(input_ids_all_padded[2])
    print(input_ids_all_padded[3])

    print(label_ids_all_padded[0])
    print(label_ids_all_padded[1])
    print(label_ids_all_padded[2])
    print(label_ids_all_padded[3])

    exit()
    """

    input_ids_tensor = torch.LongTensor(input_ids_all_padded)
    label_ids_tensor = torch.LongTensor(label_ids_all_padded)
    batch = {'input_ids': input_ids_tensor, 'label_ids': label_ids_tensor}
    #batch['len_bin'] = len_bin_list
    return batch


def get_control_mode_special_ids_dict(control_modes, tokenizer):
    control_mode_special_ids_dict = {}
    for control_mode in control_modes:
        if control_mode == 1:
            special_tokens_list = ['<len_{}>'.format(i) for i in range(10)]
            # print(special_tokens_list)
            special_tokens_id_list = tokenizer.convert_tokens_to_ids(special_tokens_list)
            # print(special_tokens_id_list)
            control_mode_special_ids_dict['len_bin'] = special_tokens_id_list
        elif control_mode == 5 or control_mode == 6 or control_mode == 4:
            special_tokens_list = ['<abs_{}>'.format(i) for i in range(3)]
            special_tokens_id_list = tokenizer.convert_tokens_to_ids(special_tokens_list)
            control_mode_special_ids_dict['abs_bin'] = special_tokens_id_list
    return control_mode_special_ids_dict


def train(args, train_dataset, model, tokenizer, epoch_state_dict, control_mode_special_ids_dict={}):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.continue_training:
        global_step = epoch_state_dict['global_step']
        epoch = epoch_state_dict['epoch']
    else:
        global_step = 0
        epoch = 0

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("TL;DR:"))
    #TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<control><summarize>:"))
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    pad_idx = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
    eos_idx = tokenizer.eos_token_id
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=coll_ml(TLDR_id_list=TLDR_id_list, pad_idx=pad_idx, eos_idx=eos_idx,
                                                     tokenizer=tokenizer, control_modes=args.control_modes,
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
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.continue_training:
        #print(epoch_state_dict['optimizer'])
        #print(epoch_state_dict['scheduler'])
        #exit()
        optimizer.load_state_dict(epoch_state_dict['optimizer'])
        if not args.do_not_reload_scheduler:
            scheduler.load_state_dict(epoch_state_dict['scheduler'])

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        #if args.continue_training:
        #    amp.load_state_dict(epoch_state_dict['amp'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("logging_steps = %d", args.logging_steps)

    #global_step = 0
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
            inputs = batch['input_ids']
            labels = batch['label_ids']
            #inputs, labels = batch
            # attn mask
            mask = torch.ne(inputs, pad_idx).float()
            # construct label, set all the pad_idx to -1 so that it will not be computed in the loss
            #labels = inputs.masked_fill(inputs == pad_idx, -1)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            mask = mask.to(args.device)
            model.train()
            outputs = model(inputs, attention_mask=mask, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    #print("loss grad: {}".format(model.lm_head.weight[0, 0].grad))
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                """
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)
                """

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # evaluate at the end of each epoch
        if args.local_rank in [-1, 0]:
            logging.info("Finished epoch {}".format(epoch))

            # save epoch state for continue training
            if epoch % 1 == 0:
                # save epoch state
                current_epoch_state_dir = os.path.join(args.output_dir, 'epoch_states', '{}-epoch'.format(epoch))
                if not os.path.exists(current_epoch_state_dir):
                    os.makedirs(current_epoch_state_dir)
                current_epoch_state = {
                    'epoch': epoch,
                    # 'total_batch': total_batch,
                    # 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'amp': amp.state_dict() if args.fp16 else None,
                    'global_step': global_step
                }
                epoch_state_dict_path = os.path.join(current_epoch_state_dir, 'epoch_state_dict.pt')
                torch.save(  # save epoch states
                    current_epoch_state,
                    epoch_state_dict_path
                )
                # save model
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(current_epoch_state_dir)
                logging.info("saved epoch state dict and model.")

            if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, model, tokenizer, control_mode_special_ids_dict=control_mode_special_ids_dict)
                print("validation perplexity: {}".format(results["perplexity"]))
                for key, value in results.items():
                    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        if args.max_epochs > 0 and epoch >= args.max_epochs:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", control_mode_special_ids_dict={}):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    #args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("TL;DR:"))
    #TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<control><summarize>:"))
    pad_idx = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
    eos_idx = tokenizer.eos_token_id
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=coll_ml(TLDR_id_list=TLDR_id_list, pad_idx=pad_idx, eos_idx=eos_idx,
                                                    tokenizer=tokenizer, control_modes=args.control_modes,
                                                    control_mode_special_ids_dict=control_mode_special_ids_dict,
                                                    input_trunc_len=args.input_trunc_length, output_trunc_len=args.output_trunc_length))

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        #inputs, labels = batch
        inputs = batch['input_ids']
        labels = batch['label_ids']
        # attn mask
        mask = torch.ne(inputs, pad_idx).float()
        # construct label, set all the pad_idx to -1 so that it will not be computed in the loss
        #labels = inputs.masked_fill(inputs == pad_idx, -1)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        mask = mask.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, attention_mask=mask, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result


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
    parser.add_argument('--control_modes', nargs='+', default=[], type=int,
                        help='0: nothing. 1: control length bin. 2: control exact length. 3: novel 3-gram range. 4: novel 2-gram range. 5: extractive_fragment density bin. 6: sentence fusion')

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
    parser.add_argument('--continue_training', action="store_true",
                        help="Train from a saved epoch.")
    parser.add_argument('--input_trunc_length', type=int, default=512,
                        help='Max length of output.')
    parser.add_argument('--output_trunc_length', type=int, default=100,
                        help='Max length of output.')

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
    parser.add_argument('--do_not_reload_scheduler', action="store_true",
                        help='Do not reload the state of optimization scheduler. Only effective when using continue training.')
    #parser.add_argument('--train_from', default="",
    #                   help="Train from a saved epoch.")

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

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    #if args.eval_data_file is None and args.do_eval:
    #    raise ValueError(
    #        "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
    #        "or remove the --do_eval argument.")

    if args.local_rank in [-1, 0]:
        if os.path.exists(args.output_dir) and os.listdir(
                args.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir))

        # create directory
        os.makedirs(args.output_dir)
        os.makedirs(join(args.output_dir, 'epoch_states'))

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
                        filename=join('exp', args.output_dir.split("/")[-1] + ".log"),
                        filemode='w')
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

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
    #special_tokens_dict = {'pad_token': '<pad>', 'additional_special_tokens': ['<summarize>', '<control>']}
    special_tokens_dict = {'pad_token': '<pad>', 'additional_special_tokens': []}
    # control modes special tokens
    for control_mode in args.control_modes:
        if control_mode == 1:
            for i in range(10):
                special_tokens_dict['additional_special_tokens'].append('<len_{}>'.format(i))
        elif control_mode == 5 or control_mode == 6 or control_mode == 4:
            for i in range(3):
                special_tokens_dict['additional_special_tokens'].append('<abs_{}>'.format(i))
        elif control_mode == 7:
            special_tokens_dict['additional_special_tokens'].append('<ent>')
            special_tokens_dict['additional_special_tokens'].append('<ent_end>')
    print(special_tokens_dict)
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # get ids of special tokens for different control modes
    control_mode_special_ids_dict = get_control_mode_special_ids_dict(args.control_modes, tokenizer)

    # move model to gpu
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Load epoch dict
    if args.continue_training:
        #latest_epoch_state = find_latest_epoch_state(join(args.model_name_or_path, 'epoch_states'))
        #print("Load epoch state from {}".format(latest_epoch_state))
        #epoch_state_dict = torch.load(join(args.model_name_or_path, 'epoch_states', latest_epoch_state))
        epoch_state_dict = torch.load(join(args.model_name_or_path, 'epoch_state_dict.pt'), map_location=lambda storage, loc: storage)
        logger.info("Load state dict in {}".format(args.model_name_or_path))
    else:
        epoch_state_dict = None

    # save tokenizer before training:
    tokenizer.save_pretrained(args.output_dir)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, epoch_state_dict, control_mode_special_ids_dict)
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
            result = evaluate(args, model, tokenizer, prefix=prefix, control_mode_special_ids_dict=control_mode_special_ids_dict)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()


