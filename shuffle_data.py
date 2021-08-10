from os.path import join
import json
import os
import re
import argparse
import random


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def shuffle_dataset(data_dir, out_dir, split):
    data_split_dir = join(data_dir, split)
    output_split_dir = join(out_dir, split)
    # make output split folder
    if not os.path.exists(output_split_dir):
        os.makedirs(output_split_dir)
    n_data = _count_data(data_split_dir)
    output_idx_list = list(range(n_data))
    random.shuffle(output_idx_list)

    for i in range(n_data):
        js = json.load(open(join(data_split_dir, '{}.json'.format(i))))
        with open(join(output_split_dir, '{}.json'.format(output_idx_list[i])), 'w') as f:
            json.dump(js, f, indent=4)
    return


def preprocess(args):
    # make output folder
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # process each split
    for split in ['train', 'val', 'test']:
        print("Process {}".format(split))
        shuffle_dataset(args.data_dir, args.out_dir, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('')
    )
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                                reproducibility.""")
    parser.add_argument('-data_dir', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-out_dir', type=str, action='store',
                        help='The directory of the data.')

    args = parser.parse_args()

    random.seed(args.seed)

    preprocess(args)
