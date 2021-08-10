# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import re
import json
import pickle
from cytoolz import curry
from construct_named_entity_masked_questions import check_present_named_entities
from summaqa.f1_squad import f1_score

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
}


class ClozeMCDataset(Dataset):
    def __init__(self, tokenizer, args, data_dir, split):
        split_dir = os.path.join(data_dir, split)
        cached_features_file = os.path.join(data_dir, args.model_type + '_cached_cloze_squad_' + split)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", data_dir)

            n_data = _count_data(split_dir)
            self.examples = []
            sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
            print(sep_id)

            num_discard_questions = 0
            num_positive_questions = 0
            num_negative_questions = 0

            for i in range(n_data):
                js = json.load(open(os.path.join(split_dir, '{}.json'.format(i))))
                if js['masked_question_list'] and js['paraphrase_context']:
                    masked_question_list = js['masked_question_list']
                    answer_list = js['answer_list']
                    reference_summary_sent_list = js['reference_summary_sent_list']
                    reference_summary_str = ' '.join(reference_summary_sent_list)
                    for masked_question, answer in zip(masked_question_list, answer_list):
                        input_str = '[CLS] ' + masked_question + ' [SEP] ' + reference_summary_str + ' [SEP]'
                        input_str_tokenized = tokenizer.tokenize(input_str)
                        answer_str_tokenized = tokenizer.tokenize(answer)
                        entity_start_end_list = check_present_named_entities(input_str_tokenized, [answer_str_tokenized])
                        start_position, end_position = entity_start_end_list[0][0], entity_start_end_list[0][1] - 1
                        #print(input_str)
                        #print(answer)
                        #print(input_str_tokenized)
                        #print(answer_str_tokenized)
                        #print("{}, {}".format(start_position, end_position))
                        input_ids = tokenizer.convert_tokens_to_ids(input_str_tokenized)
                        input_ids_len = len(input_ids)
                        if input_ids_len > 512:
                            num_discard_questions += 1
                            continue
                        first_segment_end_position = input_ids.index(sep_id)
                        token_type_ids = [0 if i <= first_segment_end_position else 1 for i in range(input_ids_len)]
                        #print(sep_id)
                        #print(token_type_ids)
                        self.examples.append((input_ids, token_type_ids, start_position, end_position, answer.lower()))
                        num_positive_questions += 1
                        #print()

                        # insert paraphrase context
                        paraphrase_input_str = '[CLS] ' + masked_question + ' [SEP] ' + js['paraphrase_context'] + ' [SEP]'
                        paraphrase_input_str_tokenized = tokenizer.tokenize(paraphrase_input_str)
                        entity_start_end_list = check_present_named_entities(paraphrase_input_str_tokenized,
                                                                             [answer_str_tokenized])
                        start_position, end_position = entity_start_end_list[0][0], entity_start_end_list[0][1] - 1
                        paraphrase_input_ids = tokenizer.convert_tokens_to_ids(paraphrase_input_str_tokenized)
                        paraphrase_input_ids_len = len(paraphrase_input_ids)
                        if paraphrase_input_ids_len > 512:
                            num_discard_questions += 1
                            continue
                        first_segment_end_position = paraphrase_input_ids.index(sep_id)
                        paraphrase_token_type_ids = [0 if i <= first_segment_end_position else 1 for i in range(paraphrase_input_ids_len)]
                        self.examples.append((paraphrase_input_ids, paraphrase_token_type_ids, start_position, end_position, answer.lower()))
                        num_positive_questions += 1
                        #print(paraphrase_input_str)
                        #print(answer)
                        #print(paraphrase_input_str_tokenized)
                        #print(answer_str_tokenized)
                        #print("{}, {}".format(start_position, end_position))
                        #print(sep_id)
                        #print(paraphrase_input_ids)
                        #print(paraphrase_token_type_ids)
                        #exit()

                    # negative questions
                    for negative_masked_question, negative_context in zip(js['negative_masked_question_list'], js['negative_context_list']):
                        input_str = '[CLS] ' + negative_masked_question + ' [SEP] ' + negative_context + ' [SEP]'
                        input_str_tokenized = tokenizer.tokenize(input_str)
                        start_position, end_position = 0, 0
                        # print(input_str)
                        # print(answer)
                        # print(input_str_tokenized)
                        # print(answer_str_tokenized)
                        # print("{}, {}".format(start_position, end_position))
                        input_ids = tokenizer.convert_tokens_to_ids(input_str_tokenized)
                        input_ids_len = len(input_ids)
                        if input_ids_len > 512:
                            num_discard_questions += 1
                            continue
                        first_segment_end_position = input_ids.index(sep_id)
                        token_type_ids = [0 if i <= first_segment_end_position else 1 for i in range(input_ids_len)]
                        answer = "[CLS]"
                        # print(sep_id)
                        # print(token_type_ids)
                        self.examples.append((input_ids, token_type_ids, start_position, end_position, answer))
                        # print()
                        num_negative_questions +=1

            print("# discard questions: {}".format(num_discard_questions))
            print("# positive questions: {}".format(num_positive_questions))
            print("# negative questions: {}".format(num_negative_questions))

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


@curry
def coll_ml(batch, tokenizer):
    #self.examples.append((input_ids, token_type_ids, start_position, end_position, answer))
    input_ids_list, token_type_ids_list, start_position_list, end_position_list, answer_list = list(zip(*batch))
    seq_lens = [len(input_ids) for input_ids in input_ids_list]
    max_seq_len = max(seq_lens)
    input_ids_list_padded = []
    token_type_ids_list_padded = []

    for input_ids, token_type_ids, start_position, end_position, _ in batch:
        input_seq_len = len(input_ids)
        padding_length = max_seq_len - input_seq_len
        input_ids_list_padded.append( input_ids + [tokenizer.pad_token_id] * padding_length )
        token_type_ids_list_padded.append( token_type_ids + [1] * padding_length )
    input_ids_tensor = torch.LongTensor(input_ids_list_padded)
    token_type_ids_tensor = torch.LongTensor(token_type_ids_list_padded)
    start_position_tensor = torch.LongTensor(start_position_list)
    end_position_tensor = torch.LongTensor(end_position_list)

    output_batch = {'input_ids': input_ids_tensor, 'token_type_ids': token_type_ids_tensor,
                    'start_positions': start_position_tensor, 'end_positions': end_position_tensor,
                    'answers': list(answer_list)}
    return output_batch


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=coll_ml(tokenizer=tokenizer))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_f1 = 0.0
    best_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            input_ids = batch['input_ids']
            token_type_ids = batch['token_type_ids']
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
            mask = torch.ne(input_ids, tokenizer.pad_token_id).float()

            input_ids = input_ids.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            start_positions = start_positions.to(args.device)
            end_positions = end_positions.to(args.device)
            mask = mask.to(args.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "token_type_ids": token_type_ids,
                "start_positions": start_positions,
                "end_positions": end_positions
            }
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if results["eval_f1"] > best_dev_f1:
                            best_dev_f1 = results["eval_f1"]
                            best_steps = global_step
                            if args.do_test:
                                results_test = evaluate(args, model, tokenizer, test=True)
                                for key, value in results_test.items():
                                    tb_writer.add_scalar("test_{}".format(key), value, global_step)
                                logger.info(
                                    "test f1: %s, global steps: %s",
                                    str(results_test["eval_f1"]),
                                    #str(results_test["eval_loss"]),
                                    str(global_step),
                                )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, model, tokenizer, prefix="", test=False):
    #eval_task_names = (args.task_name,)
    #eval_outputs_dirs = (args.output_dir,)
    eval_output_dir = args.output_dir

    results = {}
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=not test, test=test)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=coll_ml(tokenizer=tokenizer))

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    pred_answer_list = []
    true_answer_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        #batch = tuple(t.to(args.device) for t in batch)
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        answers = batch['answers']
        #start_positions = batch['start_positions']
        #end_positions = batch['end_positions']
        mask = torch.ne(input_ids, tokenizer.pad_token_id).float()

        input_ids = input_ids.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        #start_positions = start_positions.to(args.device)
        #end_positions = end_positions.to(args.device)
        mask = mask.to(args.device)

        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "token_type_ids": token_type_ids
            }
            outputs = model(**inputs)
            start_logits, end_logits = outputs

        nb_eval_steps += 1

        # start_values [batch, 1], start_indices [batch, 1]
        _, start_indices = start_logits.topk(k=1, dim=1)
        # end_values [batch, 1], end_indices [batch, 1]
        _, end_indices = end_logits.topk(k=1, dim=1)

        true_answer_list += answers
        input_ids_list = input_ids.tolist()
        for i, input_ids in enumerate(input_ids_list):
            pred_answer_ids = input_ids[start_indices[i][0]: end_indices[i][0] + 1]
            pred_answer = tokenizer.decode(pred_answer_ids, clean_up_tokenization_spaces=False)
            pred_answer_list.append(pred_answer)

    score_f_sum = 0
    for pred_answer, true_answer in zip(pred_answer_list, true_answer_list):
        f1 = f1_score(pred_answer, true_answer)
        score_f_sum += f1

    avg_fscore = score_f_sum / len(true_answer_list)
    result = {"eval_f1": avg_fscore}
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
        writer.write("model           =%s\n" % str(args.model_name_or_path))
        writer.write(
            "total batch size=%d\n"
            % (
                args.per_gpu_train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
            )
        )
        writer.write("train num epochs=%d\n" % args.num_train_epochs)
        writer.write("fp16            =%s\n" % args.fp16)
        writer.write("max seq length  =%d\n" % args.max_seq_length)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, test=False):
    dataset = ClozeMCDataset(tokenizer, args, data_dir=args.data_dir, split='val' if evaluate else 'train')
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    #parser.add_argument(
    #    "--task_name",
    #    default=None,
    #    type=str,
    #    required=True,
    #    help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    #)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=509,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

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
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # add pad token to special token
    special_tokens_dict = {'additional_special_tokens': []}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        # if args.eval_all_checkpoints: # can not use this to do test!!
        #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, test=True)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    if best_steps:
        logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)
    return results


if __name__ == "__main__":
    main()
