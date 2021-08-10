"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose

from metric import compute_rouge_l
import argparse
import re


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def get_extract_label(art_sents, abs_sents, ROUGE_mode):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        #rouges = list(map(compute_rouge_l(reference=abst, mode='r'), art_sents))
        rouges = list(map(compute_rouge_l(reference=abst, mode=ROUGE_mode), art_sents))  # Rouge-L F1
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores

@curry
def process(split, ROUGE_mode, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    tokenize = compose(list, _split_words)
    article = data['article']
    abstract = data['abstract']
    article_lower = [sent.lower() for sent in data['article']]
    abstract_lower = [sent.lower() for sent in data['abstract']]
    #art_sents = tokenize(data['article'])
    #abs_sents = tokenize(data['abstract'])
    art_sents = tokenize(article_lower)
    abs_sents = tokenize(abstract_lower)
    if art_sents and abs_sents: # some data contains empty article/abstract
        extracted, scores = get_extract_label(art_sents, abs_sents, ROUGE_mode)
    else:
        extracted, scores = [], []
    data['extracted'] = extracted
    data['score'] = scores
    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4)

def label_mp(split, ROUGE_mode):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split, ROUGE_mode),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))

def label(split, ROUGE_mode):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        tokenize = compose(list, _split_words)
        art_sents = tokenize(data['article'])
        abs_sents = tokenize(data['abstract'])
        extracted, scores = get_extract_label(art_sents, abs_sents, ROUGE_mode)
        data['extracted'] = extracted
        data['score'] = scores
        with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(data, f, indent=4)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(split, ROUGE_mode):
    if split == 'all':
        for split in ['val', 'train']:  # no need of extraction label when testing
            label_mp(split, ROUGE_mode)
    else:
        label_mp(split, ROUGE_mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make extraction laels')
    )
    parser.add_argument('--data_dir', type=str, action='store', default='all',
                        help='The path to data dir.')
    parser.add_argument('--folder_name', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    parser.add_argument('--ROUGE_mode', type=str, action='store', default='r', choices=['r', 'f'],
                        help='The metric used to construct proxy extractive target label. r means Rouge-l recall. f means ROUGE-l F1.')
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    main(args.folder_name, args.ROUGE_mode)
