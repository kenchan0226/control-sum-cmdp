import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp
from cytoolz import concat, curry, compose
import argparse
import re
from utils.string_helper import _make_n_gram
from collections import Counter
from metric import compute_rouge_l


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def _split_words(texts):
    return map(lambda t: t.lower().split(), texts)


def compute_salient_context_sent_ids(doc_sent_list_tokenized, summary_sent_list_tokenized):
    #print(summary_sent_list_tokenized)
    salient_context_sent_ids_set = set()
    for summary_word_list in summary_sent_list_tokenized:
        #rouges = list(map(compute_rouge_l(reference=abst, mode='r'), art_sents))
        rouges = list(map(compute_rouge_l(reference=summary_word_list, mode='r'), doc_sent_list_tokenized))  # Rouge-L F1
        #print(rouges)
        for doc_sent_id, rouge in enumerate(rouges):
            if rouge > 0.6:
                salient_context_sent_ids_set.add(doc_sent_id)
        #print(salient_context_sent_ids_set)
    salient_context_sent_ids_list = list(salient_context_sent_ids_set)
    salient_context_sent_ids_list.sort()
    #print(salient_context_sent_ids_list)
    #exit()
    return salient_context_sent_ids_list


@curry
def process(data_dir, i):
    try:
        with open(join(data_dir, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        doc_sent_list = js['article']
        summary_sent_list = js['abstract']

        if doc_sent_list and summary_sent_list:
            tokenize = compose(list, _split_words)
            doc_sent_list_tokenized = tokenize(doc_sent_list)
            summary_sent_list_tokenized = tokenize(summary_sent_list)
            salient_context_sent_ids = compute_salient_context_sent_ids(doc_sent_list_tokenized, summary_sent_list_tokenized)
            js['salient_context_sent_ids'] = salient_context_sent_ids
            with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
                json.dump(js, f, indent=4)
    except:
        print("json {} failed".format(i))


def label_mp(data, split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(data, split)
    n_data = _count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(data_dir),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def label(data, split):
    """ process the data split with multi-processing"""
    data_dir = join(data, split)
    n_data = _count_data(data_dir)
    for i in range(n_data):
        process(data_dir, i)


def main(data, split):
    if split == 'all':
        for split in ['val', 'train', 'test']:
            label_mp(data, split)
    else:
        label_mp(data, split)
        #label(data, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('')
    )
    parser.add_argument('-data', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    args = parser.parse_args()
    main(args.data, args.split)

