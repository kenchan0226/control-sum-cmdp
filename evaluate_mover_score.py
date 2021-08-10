import argparse
import json
import os
from os.path import join, exists
from moverscore_v2 import get_idf_dict, word_mover_score
import time
import numpy as np
import re
import torch


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.ref')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def _read_file(filename):
    # print(dec_fname)
    summary_sent_list_lower = []
    with open(filename) as f:
        for _, l in enumerate(f):
            summary_sent_list_lower.append(l.strip().lower())
    summary_str_lower = ' '.join(summary_sent_list_lower)
    return summary_str_lower


def _construct_list(dec_dir, ref_dir):
    print(dec_dir)
    print(ref_dir)
    n_data = _count_data(ref_dir)
    output_summary_str_list = []
    ref_summary_str_list = []
    for i in range(n_data):
        dec_fname = join(dec_dir, '{}.dec'.format(i))
        output_summary_str_lower = _read_file(dec_fname)
        output_summary_str_list.append(output_summary_str_lower)
        ref_fname = join(ref_dir, '{}.ref'.format(i))
        ref_summary_str_lower = _read_file(ref_fname)
        ref_summary_str_list.append(ref_summary_str_lower)
    return output_summary_str_list, ref_summary_str_list


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser('Calculate Mover Score')
    parser.add_argument('--decode_dir', action='store', required=True, help='directory of decoded summaries')
    parser.add_argument('--data', action='store', required=True, help='directory of decoded summaries')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size (default: 128)')

    args = parser.parse_args()

    start_time = time.time()

    dec_dir = join(args.decode_dir, 'output')
    with open(join(args.decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    ref_dir = join(args.data, 'refs', split)
    print(ref_dir)
    assert exists(ref_dir)

    output_summary_str_list, ref_summary_str_list = _construct_list(dec_dir, ref_dir)
    idf_dict_hyp = get_idf_dict(output_summary_str_list)  # idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = get_idf_dict(ref_summary_str_list)  # idf_dict_ref = defaultdict(lambda: 1.)

    scores = word_mover_score(ref_summary_str_list, output_summary_str_list, idf_dict_ref, idf_dict_hyp, \
                              stop_words=[], n_gram=1, remove_subwords=True, batch_size=args.batch_size)
    scores = np.array(scores)
    avg_scores = scores.mean()
    #avg_scores = np.array(scores).mean()

    print("Average word mover score: {:.5}".format(avg_scores))

    with open(join(args.decode_dir, 'moverscore.txt'), 'w') as f:
        for ms in scores:
            f.write("{:.6f}\n".format(ms))

    print("Processing time: {}s".format(time.time() - start_time))

if __name__ == "__main__":
    main()
