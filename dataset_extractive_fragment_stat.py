import os
import json
from os.path import join
import re
import argparse
import numpy as np
#import textstat
#from summaqa.summaqa import QG_masked
#from utils.cost import WordBasedDifficulty, WikiRelativeWordFrequency
import pickle as pkl
from utils.string_helper import _make_n_gram
from collections import Counter


def _make_n_gram_phrase(sequence, n=2):
    return (' '.join(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def count_matched(phrase_list, phrase_dict):
    num_matched = 0
    for phrase in phrase_list:
        if phrase in phrase_dict:
            num_matched += 1
    return num_matched


def compute_extractive_fragment(A, S):
    """
    :param A: word list of article
    :param S: word list of summary
    :return: F: a list of word list, each word list is an extractive fragment
    """
    F = []
    i = 0
    j = 0
    while i < len(S):
        f = []
        while j < len(A):
            if S[i] == A[j]:
                i_pie = i
                j_pie = j
                while S[i_pie] == A[j_pie]:
                    i_pie += 1
                    j_pie += 1
                    if i_pie >= len(S) or j_pie >= len(A):
                        break
                if len(f) <= i_pie - i:
                    f = S[i:i_pie]
                j = j_pie
            else:
                j += 1
        i = i + max(len(f), 1)
        j = 1
        if len(f) > 0:
            F.append(f)
    return F


def compute_extractive_fragment_density(ext_fragment_list, S):
    #ext_fragment_list = compute_extractive_fragment(A, S)
    density = sum([len(f)**2 for f in ext_fragment_list]) / len(S)
    return density


def compute_extractive_fragment_coverage(ext_fragment_list, S):
    #ext_fragment_list = compute_extractive_fragment(A, S)
    coverage = sum([len(f) for f in ext_fragment_list]) / len(S)
    return coverage


def n_gram_novelty(pred_n_gram_counter, src_n_gram_counter):
    all_pred_n_grams = pred_n_gram_counter.keys()
    num_unique_pred_n_grams = len(all_pred_n_grams)
    num_unique_novel_pred_n_grams = 0
    num_pred_n_grams = sum(pred_n_gram_counter.values())
    num_novel_pred_n_grams = 0
    for n_gram, cnt in pred_n_gram_counter.items():
        if n_gram not in src_n_gram_counter:
            num_unique_novel_pred_n_grams += 1
            num_novel_pred_n_grams += cnt
    unique_novel_n_gram_fraction = num_unique_novel_pred_n_grams / num_unique_pred_n_grams
    novel_n_gram_fraction = num_novel_pred_n_grams / num_pred_n_grams
    return unique_novel_n_gram_fraction, novel_n_gram_fraction


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

def main(data_dir, split):
    split_dir = join(data_dir, split)
    n_data = _count_data(split_dir)

    total_num_samples = 0
    total_num_summary_tokens = 0
    total_num_summary_sents = 0
    total_num_doc_tokens = 0
    total_num_doc_sents = 0

    all_ext_frag_density = []
    all_ext_frag_coverage = []
    abs_bin_counter = np.array([0, 0, 0])

    for i in range(n_data):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))
        if js['article'] and js['abstract']:
            total_num_samples += 1
            doc_sent_list = js['article']
            num_doc_sents = len(doc_sent_list)
            doc_str = ' '.join(doc_sent_list).lower()
            doc_word_list = doc_str.split(' ')
            num_doc_tokens = len(doc_word_list)

            summary_sent_list = js['abstract']
            num_summary_sents = len(summary_sent_list)
            summary_str = ' '.join(summary_sent_list).lower()
            summary_word_list = summary_str.split(' ')
            num_summary_tokens = len(summary_word_list)

            #ext_fragment_list = compute_extractive_fragment(doc_word_list, summary_word_list)

            all_ext_frag_density.append(float(js["extractive_fragment_density"]))

            all_ext_frag_coverage.append(float(js["extractive_fragment_coverage"]))

            total_num_summary_tokens += num_summary_tokens
            total_num_doc_tokens += num_doc_tokens
            total_num_summary_sents += num_summary_sents
            total_num_doc_sents += num_doc_sents

            # assign to bin
            if js["extractive_fragment_density"] <= 1.3:
                abs_bin = 2
            elif js["extractive_fragment_density"] <= 3.3:
                abs_bin = 1
            else:
                abs_bin = 0
            abs_bin_counter[abs_bin] += 1
            """
            if js["extractive_fragment_density"] <= 2.0:
                abs_bin = 1
            else:
                abs_bin = 0
            abs_bin_counter[abs_bin] += 1
            """

    avg_summary_tokens = total_num_summary_tokens/total_num_samples

    print("avg # tokens in summary:\t{:.3f}".format(avg_summary_tokens))
    print("avg # tokens in document:\t{:.3f}".format(total_num_doc_tokens/total_num_samples))
    print("avg # sentences in summary:\t{:.3f}".format(total_num_summary_sents/total_num_samples))
    print("avg ext. frag. density:\t{:.3f}".format(sum(all_ext_frag_density)/total_num_samples))

    print("abs bin count")
    print(abs_bin_counter)
    abs_bin_counter_normalized = abs_bin_counter / total_num_samples
    print(abs_bin_counter_normalized)

    #print("extractive fragment density")
    #all_ext_frag_density = np.array(all_ext_frag_density)
    #hist, bins = np.histogram(all_ext_frag_density, bins=[0, 1.5, 2.5, 100], density=False)
    #print(bins)
    #print(hist)
    #print("extractive fragment coverage")
    #hist, bins = np.histogram(all_ext_frag_density, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10], density=False)
    #print(bins)
    #print(hist)

    # save all_ext_frag_density array to bin
    all_ext_frag_density = np.array(all_ext_frag_density)
    with open(join(data_dir, 'ext_frag_density_{}.pkl'.format(split)), 'wb') as f:
        pkl.dump(all_ext_frag_density, f, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess review data')
    )
    parser.add_argument('-data_dir', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store',
                        help='train or val or test.')

    args = parser.parse_args()

    main(args.data_dir, args.split)
