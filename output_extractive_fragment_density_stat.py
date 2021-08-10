import os
from os.path import join
import argparse
import re
from collections import Counter, defaultdict
import numpy as np
import json
from utils.io import LEN_BINS, LEN_BINS_RANGE, ext_frag_density_to_bin
from dataset_extractive_fragment_stat import compute_extractive_fragment, compute_extractive_fragment_density


PREPOSITION = ['a', 'an', 'the']
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.dec')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def count_lines_and_tokens(dec_fname, data_fname, target_abs_bin=-1, multi_ref=False):
    num_tokens = 0
    word_list_concated = []
    word_list_sent_tokenized = []
    num_sentences = 0
    with open(dec_fname) as f:
        for i, l in enumerate(f):
            l_tokenized = l.strip().lower().split(" ")
            word_list_concated += l_tokenized
            word_list_sent_tokenized.append(l_tokenized)
            num_tokens += len(l_tokenized)
            num_sentences += 1

    output_summary_word_list = word_list_concated

    js = json.load(open(data_fname))
    doc_sent_list = js['article']
    doc_str = ' '.join(doc_sent_list).lower()
    doc_word_list = doc_str.split(' ')
    doc_word_list_sent_tokenized = [doc_sent.lower().split(' ') for doc_sent in doc_sent_list]
    #if multi_ref:
    #    reference_summary_sent_list = js['abstract'][0]
    #else:
    #    reference_summary_sent_list = js['abstract']
    reference_summary_sent_list = js['abstract']
    reference_summary_str = ' '.join(reference_summary_sent_list)
    reference_word_list = reference_summary_str.lower().split(' ')
    reference_summary_len = len(reference_word_list)
    reference_word_list_sent_tokenized = [ref_sent.lower().split(' ') for ref_sent in reference_summary_sent_list]

    if len(word_list_concated) > 0:
        if len(word_list_concated) >= 2 and word_list_concated[-2] in PREPOSITION:
            end_with_preposition = 1
        else:
            end_with_preposition = 0

        # compute ext frag density
        output_ext_fragment_list = compute_extractive_fragment(doc_word_list, output_summary_word_list)
        output_ext_frag_density = compute_extractive_fragment_density(output_ext_fragment_list,
                                                                      output_summary_word_list)
        ref_ext_frag_density = float(js['extractive_fragment_density'])

        # find abs bin
        ref_abs_bin = ext_frag_density_to_bin(ref_ext_frag_density)
        output_abs_bin = ext_frag_density_to_bin(output_ext_frag_density)

        if ref_abs_bin == output_abs_bin:
            in_reference_bin = 1
        else:
            in_reference_bin = 0

        if target_abs_bin >= 0:
            if target_abs_bin == output_abs_bin:
                in_target_bin = 1
            else:
                in_target_bin = 0
    else:
        end_with_preposition = 0
        in_reference_bin = 0
        if target_abs_bin >= 0:
            in_target_bin = 0

    stat = {}
    stat['num_sentences'] = num_sentences
    stat['num_tokens'] = num_tokens
    stat['end_with_preposition'] = end_with_preposition
    stat['in_reference_bin'] = in_reference_bin
    if target_abs_bin >= 0:
        stat['in_target_bin'] = in_target_bin

    return stat


def main(args):
    output_path = join(args.decode_dir, "output")
    n_output = _count_data(output_path)
    total_num_sentences = 0
    total_num_tokens = 0
    num_tokens_list = []
    end_with_preposition_sum = 0
    total_in_reference_bin = 0
    total_in_target_bin = 0
    in_trg_ext_frag_bin_all = []
    split_dir = join(args.data_dir, args.split)

    for i in range(n_output):  # iterate every .dec
        dec_file_path = join(output_path, "{}.dec".format(i))
        data_file_path = join(split_dir, "{}.json".format(i))
        stat = count_lines_and_tokens(dec_file_path, data_file_path, args.target_abs_bin, args.multi_ref)
        total_num_sentences += stat['num_sentences']
        total_num_tokens += stat['num_tokens']
        num_tokens_list.append(stat['num_tokens'])
        end_with_preposition_sum += stat['end_with_preposition']
        total_in_reference_bin += stat['in_reference_bin']
        if args.target_abs_bin >= 0:
            total_in_target_bin += stat['in_target_bin']
            in_trg_ext_frag_bin_all.append(stat['in_target_bin'])

    #print("average generated sentences: {:.3f}".format(total_num_sentences/n_output))
    #print("average number of tokens per summary: {:.3f}".format(total_num_tokens / n_output))
    #print("average tokens per sentence: {:.3f}".format(total_num_tokens/total_num_sentences))
    if args.target_abs_bin >= 0:
        print("average in target bin: {:.4f}".format(total_in_target_bin / n_output))
        with open(join(args.decode_dir, 'in_trg_ext_frag_bins.txt'), 'w') as f:
            for in_trg_ext_frag_bin in in_trg_ext_frag_bin_all:
                f.write("{}\n".format(in_trg_ext_frag_bin))
    else:
        print("average in reference bin: {:.4f}".format(total_in_reference_bin / n_output))
    #num_tokens_array = np.array(num_tokens_list)
    #print("min tokens: {}".format(num_tokens_array.min()))
    #print("max tokens: {}".format(num_tokens_array.max()))
    #print("std of tokens: {}".format(np.std(num_tokens_array)))
    #print("end with preposition: {}".format(end_with_preposition_sum))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Output statistics')

    # choose metric to evaluate
    parser.add_argument('-decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('-data_dir', action='store', required=True,
                        help='directory of data')
    parser.add_argument('-split', action='store', required=True,
                        help='split')
    parser.add_argument('-target_abs_bin', type=int, default=-1,
                        help='target length bin of summary (if any)')
    parser.add_argument('-multi_ref', action='store_true',
                        help='split')

    args = parser.parse_args()
    main(args)
