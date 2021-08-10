""" make reference text files needed for ROUGE evaluation """
""" Adapted from https://github.com/ChenRocks/fast_abs_rl """

import json
import os
from os.path import join, exists
from time import time
from datetime import timedelta

from utils import io
import argparse


def dump(split, data_dir):
    start = time()
    print('start processing {} split...'.format(split))
    split_dir = join(data_dir, split)
    dump_dir = join(data_dir, 'refs', split)
    n_data = io.count_data(split_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(split_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        abs_sents_list = data['abstract']
        for j, abs_sents in enumerate(abs_sents_list):
            with open(join(dump_dir, '{}.{}.ref'.format(chr(65+j), i)), 'w') as f:
                f.write(io.make_html_safe('\n'.join(abs_sents)))
    print('finished in {}'.format(timedelta(seconds=time()-start)))

def main(split, data_dir):
    if split == 'all':
        for split in ['val', 'test']:  # evaluation of train data takes too long
            if not exists(join(data_dir, 'refs', split)):
                os.makedirs(join(data_dir, 'refs', split))
            dump(split, data_dir)
    else:
        if not exists(join(data_dir, 'refs', split)):
            os.makedirs(join(data_dir, 'refs', split))
        dump(split, data_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make evaluation reference.')
    )
    parser.add_argument('-data', type=str, action='store', required=True,
                        help='The path of the data directory.')
    parser.add_argument('-split', type=str, action='store', default='all',
                        help='The folder name that needs to produce reference. all means process both val and test.')
    args = parser.parse_args()

    main(args.split, args.data)
