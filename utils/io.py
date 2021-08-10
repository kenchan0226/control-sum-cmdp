# -*- coding: utf-8 -*-
"""
Python File Template
Built on the source code of https://github.com/memray/seq2seq-keyphrase-pytorch and https://github.com/ChenRocks/fast_abs_rl
"""
#import inspect
import json
import re
import numpy as np
import os
import logging
from os.path import join
from cytoolz import curry, concat, compose
from os.path import basename
import shutil

import gensim

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from toolz.sandbox import unzip

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD = 0
UNK = 1
BOS = 2
EOS = 3
LEN_BINS = []
LEN_BINS_RANGE = [(0,33), (33,38), (38,42), (42,46), (46,50), (50,54), (54,59), (59,64), (64,72), (72,120)]
# for CNN only
for i in range(2000):
    if i <= 33:
        len_bin = 0
    elif i <= 38:
        len_bin = 1
    elif i <= 42:
        len_bin = 2
    elif i <= 46:
        len_bin = 3
    elif i <= 50:
        len_bin = 4
    elif i <= 54:
        len_bin = 5
    elif i <= 59:
        len_bin = 6
    elif i <= 64:
        len_bin = 7
    elif i <= 72:
        len_bin = 8
    else:
        len_bin = 9
    LEN_BINS.append(len_bin)
LEN_WORD = []
for i in range(10):  # 10 length bins
    word = "<len_{}>".format(i)
    LEN_WORD.append(word)
EXACT_LEN_WORD = '<len>'
ABS_BIN_WORD = ['<abs_0>', '<abs_1>', '<abs_2>']
ENTITY_SEGMENT_WORD = '<ent>'
ENTITY_END_WORD = '<ent_end>'

def n_gram_novelty_to_bin(n_gram_novelty):
    if n_gram_novelty <= 0.33333:
        bin = 0
    elif n_gram_novelty <= 0.66667:
        bin = 1
    else:
        bin = 2
    return bin

def ext_frag_density_to_bin(ext_frag_density):
    if ext_frag_density <= 1.3:
        abs_bin = 2
    elif ext_frag_density <= 3.3:
        abs_bin = 1
    else:
        abs_bin = 0
    return abs_bin

def ext_frag_density_to_bin_2bin_backup(ext_frag_density):
    if ext_frag_density <= 2.0:
        bin = 1
    else:
        bin = 0
    return bin

def ext_frag_density_to_bin_backup(ext_frag_density):
    if ext_frag_density <= 1.5:
        bin = 0
    elif ext_frag_density <= 3.0:
        bin = 1
    else:
        bin = 2
    return bin

def fusion_ratio_to_bin(fusion_ratio):
    if fusion_ratio <= 1.5:
        bin = 0
    elif fusion_ratio <= 2:
        bin = 1
    else:
        bin = 2
    return bin

class BertSeqClassifyDataset(TensorDataset):
    def __init__(self, split: str, path: str) -> None:
        self._data_pt_path = join(path, "{}.pt".format(split))
        features = torch.load(self._data_pt_path)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        super().__init__(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def get_all_labels_list(self):
        return self.tensors[3].view(-1).tolist()


def convert_to_bert_tensor(input_list):
    # input_list: a list of list of words, should not include the EOS word.
    input_list_lens = [len(l) for l in input_list]
    max_seq_len = max(input_list_lens)
    padded_batch = PAD * np.ones((len(input_list), max_seq_len))
    #segment_ids = PAD * np.ones((len(input_list), max_seq_len))

    for j in range(len(input_list)):
        current_len = input_list_lens[j]
        padded_batch[j][:current_len] = input_list[j]
        #segment_ids[j][:current_len] = 1

    padded_batch = torch.LongTensor(padded_batch)
    return padded_batch, input_list_lens


class JsonDataset(Dataset):
    def __init__(self, split: str, path: str) -> None:
        #assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())
        return js


class JsonDatasetFromIdx(Dataset):
    def __init__(self, split: str, path: str, start_idx: int) -> None:
        #assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = count_data(self._data_path) - start_idx
        self.start_idx = start_idx

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, '{}.json'.format(i + self.start_idx))) as f:
            js = json.loads(f.read())
        return js


class Many2ManyDataset(JsonDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, path):
        super().__init__(split, path)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        return js_data['article'], js_data['abstract']
        #article_string = ' '.join(js_data['article'])
        #abstract_string = ' '.join(js_data['abstract'])
        #return article_string, abstract_string


class Many2ManyDatasetWithStyle(JsonDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, path):
        super().__init__(split, path)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        if 'news_src' in js_data:
            news_src = js_data['news_src']
        else:
            news_src = None
        return js_data['article'], js_data['abstract'], news_src


class Many2ManyDatasetWithAttributes(JsonDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, path, control_modes=[]):
        super().__init__(split, path)
        self.control_modes = control_modes
        self.require_abstractiveness = False
        self.abstractiveness_field = None
        self.require_reference_entities = False
        self.require_document_entities = False
        self.require_news_src = False

        if 3 in control_modes:
            self.abstractiveness_field = 'three_gram_novelty'
            self.require_abstractiveness = True
        elif 4 in control_modes:
            self.abstractiveness_field = 'two_gram_novelty'
            self.require_abstractiveness = True
        elif 5 in control_modes:
            self.abstractiveness_field = 'extractive_fragment_density'
            self.require_abstractiveness = True
        elif 6 in control_modes:
            self.abstractiveness_field = 'avg_fusion_ratio'
            self.require_abstractiveness = True
        elif 7 in control_modes or 8 in control_modes:
            self.require_reference_entities = True

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        sample = {"src_sent_list": js_data['article'], "trg_sent_list": js_data['abstract']}
        if self.require_news_src:
            sample['news_src'] = js_data['news_src']
        if self.require_abstractiveness:
            abstractiveness = js_data[self.abstractiveness_field]
            sample['abstractiveness'] = abstractiveness
        if self.require_reference_entities:
            sample['reference_entities'] = js_data['reference_entity_list_non_numerical']
            sample['reference_entities_str'] = js_data["reference_entity_list_non_numerical_str"]
            sample['reference_entity_start_end_list'] = js_data['reference_entity_start_end_list']
            sample['masked_question_ids_list'] = js_data["masked_question_ids_list"]
            sample['answer_list'] = js_data['answer_list']
            #sample['answer_idx_list'] = js_data['answer_idx_list']
            #sample['multiple_choices_ids_list'] = js_data['multiple_choices_ids_list']
        return sample


class DecodeDataset(JsonDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, path):
        assert 'train' not in split
        super().__init__(split, path)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        #article_string = ' '.join(js_data['article'])
        return art_sents


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


@curry
def tokenize(input_str, max_len=-1):
    input_tokenized = input_str.strip().lower().split(' ')
    if max_len > 0:
        input_tokenized = input_tokenized[:max_len]
    return input_tokenized


@curry
def eval_coll_fn(batch, word2idx, src_max_len=-1):
    # batch: a list of art_str
    # tokenize article string
    source_list_tokenized = []
    for src_sent_list in batch:
        if src_sent_list:
            # join sent list into one str
            src = ' '.join(src_sent_list)
            # tokenize and truncate
            source_list_tokenized.append(tokenize(src, max_len=src_max_len))
    batch_size = len(source_list_tokenized)
    # convert to idx
    source_list_indiced = []
    source_oov_list_indiced = []
    oov_lists = []
    for src in source_list_tokenized:
        src_oov, oov_dict, oov_list = extend_vocab_oov(src, word2idx)
        src = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in src]
        source_list_indiced.append(src)
        source_oov_list_indiced.append(src_oov)
        oov_lists.append(oov_list)
    original_indices = list(range(batch_size))
    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list_indiced, source_oov_list_indiced, oov_lists, source_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list_indiced, source_oov_list_indiced, oov_lists, source_list_tokenized, original_indices = zip(*seq_pairs)
    """
    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list_indiced)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list_indiced)
    source_mask = create_padding_mask(source_tensor)

    batch_dict = {}
    batch_dict['src_tensor'] = source_tensor
    batch_dict['src_lens'] = source_lens
    batch_dict['src_mask'] = source_mask
    batch_dict['src_oov_tensor'] = source_oov_tensor
    batch_dict['oov_lists'] = oov_lists
    batch_dict['src_list_tokenized'] = source_list_tokenized
    batch_dict['original_indices'] = original_indices

    return batch_dict


@curry
def eval_coll_fn_with_style(batch, word2idx, target_style_idx, src_max_len=-1, with_len=False):
    # batch: a list of art_str
    # tokenize article string
    source_list_tokenized = []
    for src_sent_list in batch:
        if src_sent_list:
            # join sent list into one str
            src = ' '.join(src_sent_list)
            # tokenize and truncate
            source_list_tokenized.append(tokenize(src, max_len=src_max_len))
    batch_size = len(source_list_tokenized)
    # convert to idx
    source_list_indiced = []
    source_oov_list_indiced = []
    oov_lists = []
    for src in source_list_tokenized:
        src_oov, oov_dict, oov_list = extend_vocab_oov(src, word2idx)
        src = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in src]
        source_list_indiced.append(src)
        source_oov_list_indiced.append(src_oov)
        oov_lists.append(oov_list)
    original_indices = list(range(batch_size))
    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list_indiced, source_oov_list_indiced, oov_lists, source_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list_indiced, source_oov_list_indiced, oov_lists, source_list_tokenized, original_indices = zip(*seq_pairs)
    """
    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list_indiced)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list_indiced)
    source_mask = create_padding_mask(source_tensor)
    style_tensor = torch.LongTensor([target_style_idx] * batch_size)

    batch_dict = {}
    batch_dict['src_tensor'] = source_tensor
    batch_dict['src_lens'] = source_lens
    batch_dict['src_mask'] = source_mask
    batch_dict['src_oov_tensor'] = source_oov_tensor
    batch_dict['oov_lists'] = oov_lists
    batch_dict['src_list_tokenized'] = source_list_tokenized
    batch_dict['original_indices'] = original_indices
    batch_dict['style_tensor'] = style_tensor

    return batch_dict


@curry
def coll_fn(batch, word2idx, src_max_len=-1, trg_max_len=-1, with_len=False):
    # batch: a list of (art_str, abs_str)
    # Remove empty data, tokenize, and truncate
    source_list_tokenized, target_list_tokenized = [], []
    target_sent_2d_list = []
    for src_sent_list, trg_sent_list in batch:
        if src_sent_list and trg_sent_list:
            target_sent_2d_list.append(trg_sent_list)
            # concat each sent list into one str
            src = ' '.join(src_sent_list)
            trg = ' '.join(trg_sent_list)
            # tokenize and truncate
            src_tokenized = tokenize(src, src_max_len)
            trg_tokenized = tokenize(trg, trg_max_len)
            trg_len = len(trg_tokenized)
            if with_len:
                len_bin = LEN_BINS[trg_len]
                len_bin_word = LEN_WORD[len_bin]
                src_tokenized.insert(0, len_bin_word)
            source_list_tokenized.append(src_tokenized)
            target_list_tokenized.append(trg_tokenized)

    batch_size = len(source_list_tokenized)

    # convert to idx
    source_list, target_list, source_oov_list, target_oov_list, oov_lists = convert_batch_to_idx(source_list_tokenized, target_list_tokenized, word2idx)
    original_indices = list(range(batch_size))

    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices = zip(*seq_pairs)
    """

    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list)
    target_tensor, target_lens = convert_to_tensor(target_list)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list)
    target_oov_tensor, _ = convert_to_tensor(target_oov_list)
    source_mask = create_padding_mask(source_tensor)
    target_mask = create_padding_mask(target_tensor)

    batch_dict = {}
    batch_dict['src_tensor'] = source_tensor
    batch_dict['src_lens'] = source_lens
    batch_dict['src_mask'] = source_mask
    batch_dict['src_oov_tensor'] = source_oov_tensor
    batch_dict['oov_lists'] = oov_lists
    batch_dict['src_list_tokenized'] = source_list_tokenized
    batch_dict['trg_sent_2d_list'] = target_sent_2d_list
    batch_dict['trg_tensor'] = target_tensor
    batch_dict['trg_oov_tensor'] = target_oov_tensor
    batch_dict['trg_lens'] = target_lens
    batch_dict['trg_mask'] = target_mask
    batch_dict['original_indices'] = original_indices

    return batch_dict


@curry
def coll_fn_with_attribute(batch, word2idx, style_label_map=None, with_style=False, target_style_idx=-1, src_max_len=-1,
                           trg_max_len=-1, control_modes=[], with_ground_truth=False, desired_target_numbers=[],
                           is_multiple_ref=False, is_rl=False, is_testing=False):
    # batch: a list of (art_str, abs_str)
    # Remove empty data, tokenize, and truncate
    source_list_tokenized, target_list_tokenized = [], []
    target_sent_2d_list = []
    src_sent_2d_list = []
    #style_idx_list = []
    len_bin_list = []
    exact_len_list = []
    abs_bin_list = []
    position_ids_list = []
    reference_entities_list = []
    masked_questions_ids_2dlist = []
    answer_2dlist = []
    reference_entities_str_tokenized_list = []
    #answer_id_2dlist = []
    #multiple_choices_ids_2dlist = []
    if with_ground_truth:
        desired_target_numbers = [-1] * len(control_modes)

    #if control_mode > 0:
    #    assert with_ground_truth or desired_target_number >= 0

    for sample in batch:
        src_sent_list = sample['src_sent_list']
        trg_sent_list = sample['trg_sent_list']
        # src_sent_list, trg_sent_list, style, abstractiveness
        if src_sent_list:
            if not trg_sent_list and not is_testing:  # if it is training and target summary is empty, skip it
                continue
            if is_multiple_ref:
                trg_sent_list = trg_sent_list[0]
            target_sent_2d_list.append(trg_sent_list)
            src_sent_2d_list.append(src_sent_list)
            # concat each sent list into one str
            src = ' '.join(src_sent_list)
            trg = ' '.join(trg_sent_list)
            # tokenize and truncate
            src_tokenized = tokenize(src, src_max_len)
            trg_tokenized = tokenize(trg, trg_max_len)
            trg_len_without_eos = len(trg_tokenized)  # without eos token
            for control_mode, desired_target_number in zip(control_modes, desired_target_numbers):
                if control_mode == 1:
                    if with_ground_truth:
                        len_bin = LEN_BINS[trg_len_without_eos]
                    else:
                        len_bin = desired_target_number
                    len_bin_list.append(len_bin)
                    len_bin_word = LEN_WORD[len_bin]
                    src_tokenized.insert(0, len_bin_word)
                elif control_mode == 2:
                    if with_ground_truth:
                        exact_len = trg_len_without_eos
                    else:
                        exact_len = desired_target_number
                    exact_len_list.append(exact_len)
                    #src_tokenized.insert(0, EXACT_LEN_WORD)
                elif control_mode == 3 or control_mode == 4 or control_mode == 5 or control_mode == 6:
                    if with_ground_truth:
                        abstractiveness = sample['abstractiveness']
                        assert abstractiveness is not None, "If you use control mode 3 (4) to train, each sample must have the field of three_gram_novelty (two gram novelty)"
                        if control_mode == 3 or control_mode == 4:
                            abs_bin = n_gram_novelty_to_bin(abstractiveness)
                        elif control_mode == 5: # control mode 5
                            abs_bin = ext_frag_density_to_bin(abstractiveness)
                        elif control_mode == 6:
                            abs_bin = fusion_ratio_to_bin(abstractiveness)
                    else:
                        abs_bin = desired_target_number
                    abs_bin_list.append(abs_bin)
                    abs_bin_word = ABS_BIN_WORD[abs_bin]
                    src_tokenized.insert(0, abs_bin_word)
                elif control_mode == 7:
                    if with_ground_truth:
                        reference_entities = sample['reference_entities']
                        reference_entities_str = sample['reference_entities_str']
                        reference_entity_start_end_list = sample['reference_entity_start_end_list']
                        position_ids = []
                        for start, end in reference_entity_start_end_list:
                            position_ids += list(range(start, end))
                            position_ids.append(499)
                        position_ids_list.append(position_ids)
                        if is_rl:
                            masked_questions_ids_2dlist.append(sample["masked_question_ids_list"])
                            answer_2dlist.append(sample["answer_list"])
                            #answer_id_2dlist.append(sample["answer_idx_list"])
                            #multiple_choices_ids_2dlist.append(sample["multiple_choices_ids_list"])
                    else:
                        # reference_entity_str_list_non_numerical = [entity_str for entity_str, _ in reference_entities]
                        # reference_entity_list_non_numerical_str = ' <ent> '.join(reference_entities) + ' <ent_end>'
                        raise ValueError
                    # prepend to input document
                    # A [ent] B [ent] C [ent_end]
                    reference_entities_list.append(reference_entities)
                    src_tokenized = reference_entities_str.split(' ') + src_tokenized
                elif control_mode == 8:
                    reference_entities_str = sample['reference_entities_str']
                    if reference_entities_str == "":
                        reference_entities_str = "<ent_end>"
                    reference_entities_str_tokenized = tokenize(reference_entities_str, trg_max_len)
                    reference_entities_str_tokenized_list.append(reference_entities_str_tokenized)

            source_list_tokenized.append(src_tokenized)
            target_list_tokenized.append(trg_tokenized)
            #if style_label_map is not None:
            #    style_idx_list.append(style_label_map[style])
        elif is_testing:  # if it is testing and do not have source sentences
            raise ValueError

    batch_size = len(source_list_tokenized)

    # convert to idx
    source_list, target_list, source_oov_list, target_oov_list, oov_lists = convert_batch_to_idx(source_list_tokenized, target_list_tokenized, word2idx)
    original_indices = list(range(batch_size))

    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices = zip(*seq_pairs)
    """

    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list)
    target_tensor, target_lens = convert_to_tensor(target_list)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list)
    target_oov_tensor, _ = convert_to_tensor(target_oov_list)
    source_mask = create_padding_mask(source_tensor)
    target_mask = create_padding_mask(target_tensor)
    # position id tensor
    # takes no effect
    if 7 in control_modes:
        src_len = source_tensor.size(1)
        position_ids_padded = []
        #print("pad position ids")
        #print(src_len)
        for position_ids in position_ids_list:
            position_ids_len = len(position_ids)
            position_ids += range(0, src_len - position_ids_len)
            position_ids_padded.append(position_ids)
            #print(position_ids)
        position_ids_tensor = torch.LongTensor(position_ids_padded)
    else:
        position_ids_tensor = None
    if 8 in control_modes:
        query_list = convert_query_batch_to_idx(reference_entities_str_tokenized_list, word2idx)
        query_tensor, query_lens = convert_to_tensor(query_list)
        query_mask = create_padding_mask(query_tensor)
        """
        print("reference_entities_str_tokenized_list")
        print(reference_entities_str_tokenized_list)
        print()
        print("query_tensor")
        print(query_tensor)
        print()
        print("query_mask")
        print(query_mask)
        print()
        print("query_lens")
        print(query_lens)
        exit()
        """

    #print(reference_entities_list)
    batch_dict = {}
    batch_dict['src_tensor'] = source_tensor
    batch_dict['src_lens'] = source_lens
    batch_dict['src_mask'] = source_mask
    batch_dict['src_oov_tensor'] = source_oov_tensor
    batch_dict['oov_lists'] = oov_lists
    batch_dict['src_list_tokenized'] = source_list_tokenized
    batch_dict['trg_sent_2d_list'] = target_sent_2d_list
    batch_dict['trg_tensor'] = target_tensor
    batch_dict['trg_oov_tensor'] = target_oov_tensor
    batch_dict['trg_lens'] = target_lens
    batch_dict['trg_mask'] = target_mask
    batch_dict['original_indices'] = original_indices
    batch_dict['len_bins'] = len_bin_list
    batch_dict['abs_bins'] = abs_bin_list
    batch_dict['exact_lens'] = exact_len_list
    batch_dict['src_sent_2d_list'] = src_sent_2d_list
    batch_dict['masked_questions_ids_2dlist'] = masked_questions_ids_2dlist
    batch_dict['answer_2dlist'] = answer_2dlist
    #batch_dict['answer_id_2dlist'] = answer_id_2dlist
    #batch_dict['multiple_choices_ids_2dlist'] = multiple_choices_ids_2dlist
    batch_dict['reference_entities_list'] = reference_entities_list
    batch_dict['position_ids'] = position_ids_tensor
    if 8 in control_modes:
        batch_dict['query_tensor'] = query_tensor
        batch_dict['query_mask'] = query_mask
        batch_dict['query_lens'] = query_lens

    #if with_style:
    #    style_tensor = torch.LongTensor(style_idx_list)
    #elif target_style_idx >= 0:
    #    style_tensor = torch.LongTensor([target_style_idx] * batch_size)
    #else:
    #    style_tensor = None
    #batch_dict['style_tensor'] = style_tensor

    return batch_dict


def truncate_and_tokenize_sent_list(sent_list, max_len):
    current_length = 0
    sent_list_truncated_tokenized = []
    for sent in sent_list:
        sent_tokenized = sent.split(' ')
        sent_list_truncated_tokenized.append(sent_tokenized)
        current_length += len(sent_tokenized)
        if current_length >= max_len:
            break
    return sent_list_truncated_tokenized


@curry
def coll_fn_with_style_backup(batch, word2idx, style_label_map=None, with_style=False, target_style_idx=-1, src_max_len=-1, trg_max_len=-1, with_len=False, target_len_bin=-1):
    # batch: a list of (art_str, abs_str)
    # Remove empty data, tokenize, and truncate
    source_list_tokenized, target_list_tokenized = [], []
    target_sent_2d_list = []
    style_idx_list = []
    len_bin_list = []
    for src_sent_list, trg_sent_list, style in batch:
        if src_sent_list and trg_sent_list:
            target_sent_2d_list.append(trg_sent_list)
            # concat each sent list into one str
            src = ' '.join(src_sent_list)
            trg = ' '.join(trg_sent_list)
            # tokenize and truncate
            src_tokenized = tokenize(src, src_max_len)
            trg_tokenized = tokenize(trg, trg_max_len)
            trg_len = len(trg_tokenized)
            if with_len or target_len_bin >= 0:
                if with_len:
                    len_bin = LEN_BINS[trg_len]
                else:
                    len_bin = target_len_bin
                len_bin_list.append(len_bin)
                len_bin_word = LEN_WORD[len_bin]
                src_tokenized.insert(0, len_bin_word)
            source_list_tokenized.append(src_tokenized)
            target_list_tokenized.append(trg_tokenized)
            if style_label_map is not None:
                style_idx_list.append(style_label_map[style])

    batch_size = len(source_list_tokenized)

    # convert to idx
    source_list, target_list, source_oov_list, target_oov_list, oov_lists = convert_batch_to_idx(source_list_tokenized, target_list_tokenized, word2idx)
    original_indices = list(range(batch_size))

    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices = zip(*seq_pairs)
    """

    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list)
    target_tensor, target_lens = convert_to_tensor(target_list)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list)
    target_oov_tensor, _ = convert_to_tensor(target_oov_list)
    source_mask = create_padding_mask(source_tensor)
    target_mask = create_padding_mask(target_tensor)

    batch_dict = {}
    batch_dict['src_tensor'] = source_tensor
    batch_dict['src_lens'] = source_lens
    batch_dict['src_mask'] = source_mask
    batch_dict['src_oov_tensor'] = source_oov_tensor
    batch_dict['oov_lists'] = oov_lists
    batch_dict['src_list_tokenized'] = source_list_tokenized
    batch_dict['trg_sent_2d_list'] = target_sent_2d_list
    batch_dict['trg_tensor'] = target_tensor
    batch_dict['trg_oov_tensor'] = target_oov_tensor
    batch_dict['trg_lens'] = target_lens
    batch_dict['trg_mask'] = target_mask
    batch_dict['original_indices'] = original_indices
    batch_dict['len_bins'] = len_bin_list

    if with_style:
        style_tensor = torch.LongTensor(style_idx_list)
    elif target_style_idx >= 0:
        style_tensor = torch.LongTensor([target_style_idx] * batch_size)
    else:
        style_tensor = None

    batch_dict['style_tensor'] = style_tensor

    return batch_dict


@curry
def coll_fn_backup(batch, word2idx, src_max_len=-1, trg_max_len=-1):
    # batch: a list of (art_str, abs_str)
    # Remove empty data, tokenize, and truncate
    source_list_tokenized, target_list_tokenized = [], []
    target_sent_2d_list = []
    for src_sent_list, trg_sent_list in batch:
        if src_sent_list and trg_sent_list:
            target_sent_2d_list.append(trg_sent_list)
            # concat each sent list into one str
            src = ' '.join(src_sent_list)
            trg = ' '.join(trg_sent_list)
            # tokenize and truncate
            source_list_tokenized.append(tokenize(src, src_max_len))
            target_list_tokenized.append(tokenize(trg, trg_max_len))

    batch_size = len(source_list_tokenized)

    # convert to idx
    source_list, target_list, source_oov_list, target_oov_list, oov_lists = convert_batch_to_idx(source_list_tokenized, target_list_tokenized, word2idx)
    original_indices = list(range(batch_size))

    # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
    """
    seq_pairs = sorted(zip(source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices),
                       key=lambda p: len(p[0]), reverse=True)
    source_list, target_list, source_oov_list, target_oov_list, oov_lists, source_list_tokenized, target_list_tokenized, original_indices = zip(*seq_pairs)
    """

    # padding to tensor
    source_tensor, source_lens = convert_to_tensor(source_list)
    target_tensor, target_lens = convert_to_tensor(target_list)
    source_oov_tensor, _ = convert_to_tensor(source_oov_list)
    target_oov_tensor, _ = convert_to_tensor(target_oov_list)
    source_mask = create_padding_mask(source_tensor)
    target_mask = create_padding_mask(target_tensor)

    return source_tensor, source_lens, source_mask, source_oov_tensor, oov_lists, source_list_tokenized, target_sent_2d_list, target_tensor, target_oov_tensor, target_lens, target_mask, original_indices


def convert_batch_to_idx(source_list, target_list, word2idx):
    source_list_indiced, target_list_indiced, source_oov_list_indiced, target_oov_list_indiced = [], [], [], []
    oov_lists = []
    for src, trg in zip(source_list, target_list):
        src_oov, oov_dict, oov_list = extend_vocab_oov(src, word2idx)
        src = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in src]
        trg_oov = []
        for w in trg:
            if w in word2idx:
                trg_oov.append(word2idx[w])
            elif w in oov_dict:
                trg_oov.append(oov_dict[w])
            else:
                trg_oov.append(word2idx[UNK_WORD])
        trg = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in trg]
        source_list_indiced.append(src)
        target_list_indiced.append(trg)
        source_oov_list_indiced.append(src_oov)
        target_oov_list_indiced.append(trg_oov)
        oov_lists.append(oov_list)
    return source_list_indiced, target_list_indiced, source_oov_list_indiced, target_oov_list_indiced, oov_lists


def convert_query_batch_to_idx(query_list, word2idx):
    query_list_indiced = []
    for query in query_list:
        query_ids = [word2idx[w] if w in word2idx else word2idx[UNK_WORD] for w in query]
        query_list_indiced.append(query_ids)
    return query_list_indiced


def extend_vocab_oov(src_words, word2idx):
    oov_dict = {}
    src_oov = []
    for src_word in src_words:
        if src_word in word2idx:
            src_oov.append(word2idx[src_word])
        else:
            if src_word in oov_dict:
                idx = oov_dict[src_word]
            else:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                idx = len(word2idx) + len(oov_dict)
                oov_dict[src_word] = idx
            src_oov.append(idx)
    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x: x[1])]
    return src_oov, oov_dict, oov_list


def convert_to_tensor(input_list):
    # input_list: a list of list of words, should not include the EOS word.
    input_list = [l + [EOS] for l in input_list]  # append EOS at the end of each word list
    input_list_lens = [len(l) for l in input_list]
    max_seq_len = max(input_list_lens)
    padded_batch = PAD * np.ones((len(input_list), max_seq_len))

    for j in range(len(input_list)):
        current_len = input_list_lens[j]
        padded_batch[j][:current_len] = input_list[j]

    padded_batch = torch.LongTensor(padded_batch)
    return padded_batch, input_list_lens


def create_padding_mask(padded_batch):
    #pad_idx = word2idx[PAD_WORD]
    input_mask = torch.ne(padded_batch, PAD)
    input_mask = input_mask.type(torch.FloatTensor)
    return input_mask


def make_vocab_with_special_token(wc, vocab_size, control_modes):
    word2idx, idx2word = {}, {}
    word2idx[PAD_WORD] = 0
    word2idx[UNK_WORD] = 1
    word2idx[BOS_WORD] = 2
    word2idx[EOS_WORD] = 3
    idx = 3
    if 1 in control_modes:
        for i in range(10):  # 10 length bins
            idx += 1
            word2idx[LEN_WORD[i]] = idx
    if 2 in control_modes:
        idx += 1
        word2idx[EXACT_LEN_WORD] = idx
    if 3 in control_modes or 4 in control_modes or 5 in control_modes or 6 in control_modes:
        for i in range(len(ABS_BIN_WORD)):
            idx += 1
            word2idx[ABS_BIN_WORD[i]] = idx
    if 7 in control_modes or 8 in control_modes:
        idx += 1
        word2idx[ENTITY_SEGMENT_WORD] = idx
        idx += 1
        word2idx[ENTITY_END_WORD] = idx
    for i, (w, _) in enumerate(wc.most_common(vocab_size), idx + 1):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w
    return word2idx, idx2word


def make_vocab(wc, vocab_size):
    word2idx, idx2word = {}, {}
    word2idx[PAD_WORD] = 0
    word2idx[UNK_WORD] = 1
    word2idx[BOS_WORD] = 2
    word2idx[EOS_WORD] = 3
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w
    return word2idx, idx2word


def make_vocab_with_len_bin(wc, vocab_size):
    word2idx, idx2word = {}, {}
    word2idx[PAD_WORD] = 0
    word2idx[UNK_WORD] = 1
    word2idx[BOS_WORD] = 2
    word2idx[EOS_WORD] = 3
    for i in range(10):  # 10 length bins
        idx = 4 + i
        word2idx[LEN_WORD[i]] = idx

    for i, (w, _) in enumerate(wc.most_common(vocab_size), 14):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w
    return word2idx, idx2word


def make_vocab_with_exact_len_token(wc, vocab_size):
    word2idx, idx2word = {}, {}
    word2idx[PAD_WORD] = 0
    word2idx[UNK_WORD] = 1
    word2idx[BOS_WORD] = 2
    word2idx[EOS_WORD] = 3
    word2idx[EXACT_LEN_WORD] = 4

    for i, (w, _) in enumerate(wc.most_common(vocab_size), 5):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w
    return word2idx, idx2word


def make_embedding(idx2word, w2v_file):
    attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(idx2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    # initialize embedding weight first
    init_range = 0.1
    embedding.data.uniform_(-init_range, init_range)

    #if initializer is not None:
    #    initializer(embedding)
    oovs = []
    with torch.no_grad():
        for i in range(len(idx2word)):
            # NOTE: idx2word can be list or dict
            if i == BOS:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == EOS:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif idx2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[idx2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def remove_old_ckpts(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward, Only keep the highest three checkpoints. """
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    score_list = [float(ckpt.split('-')[-1]) for ckpt in ckpts]
    ckpts_score_sorted = sorted(zip(score_list, ckpts), key=lambda p: p[0], reverse=reverse)
    _, ckpts_sorted = zip(*ckpts_score_sorted)
    for ckpt in ckpts_sorted[6:]:
        os.remove(join(model_dir, 'ckpt', ckpt))
    logging.info("Best model: {}".format(join(model_dir, 'ckpt', ckpts_sorted[0])))
    #print("Best model: {}".format(join(model_dir, 'ckpt', ckpts_sorted[0])))

def remove_old_epoch_states(epoch_state_dir):
    epoch_states = os.listdir(epoch_state_dir)
    epoch_list = [int(epoch_state.split('-')[0]) for epoch_state in epoch_states]
    epoch_list.sort()
    old_epoch_list = epoch_list[:-1]
    for old_epoch in old_epoch_list:
        os.remove(join(epoch_state_dir, "{}-epoch.pt".format(old_epoch)))
    logging.info("Removed old epoch state dict.")

def find_latest_epoch_state(epoch_state_dir):
    epoch_states = os.listdir(epoch_state_dir)
    epoch_list = [int(epoch_state.split('-')[0]) for epoch_state in epoch_states]
    epoch_list.sort()
    return "{}-epoch.pt".format(epoch_list[-1])

def remove_old_ckpts_dir(model_dir, reverse=False):
    """ Only for bert classifier, reverse=False->loss, reverse=True->reward, Only keep the highest three checkpoints. """
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    score_list = [float(ckpt.split('-')[-1]) for ckpt in ckpts]
    ckpts_score_sorted = sorted(zip(score_list, ckpts), key=lambda p: p[0], reverse=reverse)
    _, ckpts_sorted = zip(*ckpts_score_sorted)
    for ckpt in ckpts_sorted[3:]:
        shutil.rmtree(join(model_dir, 'ckpt', ckpt))
        #os.remove(join(model_dir, 'ckpt', ckpt))
    logging.info("Best model: {}".format(join(model_dir, 'ckpt', ckpts_sorted[0])))
    #print("Best model: {}".format(join(model_dir, 'ckpt', ckpts_sorted[0])))

