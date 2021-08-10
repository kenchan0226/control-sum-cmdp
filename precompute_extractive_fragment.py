import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp
from cytoolz import concat, curry, compose
import argparse
import re
from dataset_extractive_fragment_stat import compute_extractive_fragment, compute_extractive_fragment_density, compute_extractive_fragment_coverage


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


@curry
def process(data_dir, i):
    try:
        with open(join(data_dir, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        doc_sent_list = js['article']
        doc_str = ' '.join(doc_sent_list).lower()
        doc_word_list = doc_str.split(' ')

        summary_sent_list = js['abstract']
        summary_str = ' '.join(summary_sent_list).lower()
        summary_word_list = summary_str.split(' ')

        if doc_sent_list and summary_sent_list:
            ext_fragment_list = compute_extractive_fragment(doc_word_list, summary_word_list)
            ext_frag_density = compute_extractive_fragment_density(ext_fragment_list, summary_word_list)
            ext_frag_coverage = compute_extractive_fragment_coverage(ext_fragment_list, summary_word_list)
        else:
            ext_frag_density, ext_frag_coverage = 0.0, 0.0

        js['extractive_fragment_density'] = ext_frag_density
        js['extractive_fragment_coverage'] = ext_frag_coverage

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


def main(data, split):
    if split == 'all':
        for split in ['val', 'train', 'test']:
            label_mp(data, split)
    else:
        label_mp(data, split)


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
