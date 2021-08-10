import torch
import config
import argparse
import pickle as pkl
from utils import io
from utils.io import DecodeDataset, eval_coll_fn
from torch.utils.data import DataLoader
import os
from os.path import join
from model.seq2seq import Seq2SeqModel
from sequence_generator import SequenceGenerator
from tqdm import tqdm
import json
from utils.string_helper import prediction_to_sentence
import nltk


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def main(pred_path, data_dir, split):

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
        os.makedirs(join(pred_path, 'output'))

    n_data = _count_data(join(data_dir, split))

    for i in range(n_data):
        js = json.load(open(join(data_dir, split, '{}.json'.format(i))))
        summary = js['abstract']

        log = {'split': 'test'}
        json.dump(log, open(join(pred_path, 'log.json'), 'w'))

        with open(join(pred_path, 'output', '{}.dec'.format(i)), 'w') as f:
            f.write(io.make_html_safe('\n'.join(summary)))


if __name__ == "__main__":
    data_dir = '../../datasets/cased-cnn-dailymail_coref_3'
    pred_path = 'pred/cnn_coref_3_gold'
    split = 'test'
    main(pred_path, data_dir, split)
