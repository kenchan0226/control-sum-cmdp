from os.path import join
import json
import os
import random
import argparse
from collections import Counter
import pickle as pkl
import re


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def main(data_dir, styles, style_weights):
    split_dir = join(data_dir, "train")
    n_data = _count_data(split_dir)
    vocab_counter = Counter()
    style_label_map = {label: i for i, label in enumerate(styles)}
    for i in range(n_data):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))

        summary = js['abstract']
        summary_text = ' '.join(summary).strip().lower()
        summary_word_list = summary_text.strip().split(' ')

        document = js['article']
        document_text = ' '.join(document).strip().lower()
        document_word_list = document_text.strip().split(' ')

        style = js['news_src']
        weight = style_weights[style_label_map[style]]

        all_tokens = summary_word_list + document_word_list
        weighted_all_tokens = weight * [t for t in all_tokens if t != ""]

        vocab_counter.update(weighted_all_tokens)

    with open(os.path.join(data_dir, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('')
    )
    parser.add_argument('-data_dir', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-styles', nargs='+', default=[], type=str,
                        help="""name of each style. """)
    parser.add_argument('-style_weights', nargs='+', default=[], type=int,
                        help="""weight for each style. """)

    args = parser.parse_args()
    main(args.data_dir, args.styles, args.style_weights)
