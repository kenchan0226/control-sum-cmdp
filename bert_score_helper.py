import os
import time
import argparse
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

from bert_score.utils import (get_idf_dict, bert_cos_score_idf,
                    get_bert_embedding, model_types,
                    lang2model, model2layers, get_hash)

from bert_score import get_model


class BertScoreReward:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.num_layers = model2layers[self.model_type]
        self.model = get_model(self.model_type, num_layers=self.num_layers, all_layers=False)
        self.device = device
        self.model.to(device)

        # do not use idf
        self.idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0

    def score(self, cands, refs):
        """
        :param cands: a list of string
        :param ref: a list of string
        :return:
        """
        batch_size = len(cands)
        all_preds = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, self.idf_dict,
                                   verbose=False, device=self.device,
                                   batch_size=batch_size, all_layers=False)
        P = all_preds[..., 0].cpu()
        R = all_preds[..., 1].cpu()
        F1 = all_preds[..., 2].cpu()
        return F1


def score(cands, refs, model_type=None, num_layers=None, verbose=False,
          idf=False, batch_size=64, nthreads=4, all_layers=False, lang=None,
          return_hash=False):
    """
    BERTScore metric.
    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str): reference sentences
        - :param: `bert` (str): bert specification
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool): use idf weighting
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences
        - :param: `return_hash` (bool): return hash code of the setting
    """
    assert len(cands) == len(refs)

    assert lang is not None or model_type is not None, \
        'Either lang or model_type should be specified'

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]


    assert model_type in model_types
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = get_model(model_type, num_layers, all_layers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    else:
        if verbose:
            print('preparing IDF dict...')
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print('done in {:.2f} seconds'.format(time.perf_counter() - start))

    if verbose:
        print('calculating scores...')
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(model, refs, cands, tokenizer, idf_dict,
                                   verbose=verbose, device=device,
                                   batch_size=batch_size, all_layers=all_layers)

    P = all_preds[..., 0].cpu()
    R = all_preds[..., 1].cpu()
    F1 = all_preds[..., 2].cpu()
    if verbose:
        time_diff = time.perf_counter() - start
        print(f'done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec')

    if return_hash:
        return (P, R, F1), get_hash(model_type, num_layers, idf)
    else:
        return P, R, F1