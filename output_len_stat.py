import os
from os.path import join
import argparse
import re
from collections import Counter, defaultdict
import numpy as np
import json
from utils.io import LEN_BINS, LEN_BINS_RANGE


PREPOSITION = ['a', 'an', 'the']
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.dec')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


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


def n_gram_repeat(pred_n_gram_counter):
    num_pred_n_grams = sum(pred_n_gram_counter.values())
    num_n_gram_repeat = sum(c - 1 for g, c in pred_n_gram_counter.items() if c > 1)
    n_gram_repeat_fraction = num_n_gram_repeat / num_pred_n_grams
    return n_gram_repeat_fraction


def count_lines_and_tokens(dec_fname, data_fname, target_len=-1, target_len_bin=-1, multi_ref=False):
    num_tokens = 0
    sum_one_gram_counter = Counter()
    sum_two_gram_counter = Counter()
    sum_three_gram_counter = Counter()
    word_list_concated = []
    num_sentences = 0
    with open(dec_fname) as f:
        for i, l in enumerate(f):
            l_tokenized = l.strip().split(" ")
            word_list_concated += l_tokenized
            sum_one_gram_counter.update(_make_n_gram(l_tokenized, n=1))
            sum_two_gram_counter.update(_make_n_gram(l_tokenized, n=2))
            sum_three_gram_counter.update(_make_n_gram(l_tokenized, n=3))
            num_tokens += len(l_tokenized)
            num_sentences += 1

    doc_one_gram_counter = Counter()
    doc_two_gram_counter = Counter()
    doc_three_gram_counter = Counter()
    js = json.load(open(data_fname))
    doc_sent_list = js['article']
    doc_str = ' '.join(doc_sent_list)
    doc_word_list = doc_str.split(' ')
    if multi_ref:
        reference_summary_sent_list = js['abstract'][0]
    else:
        reference_summary_sent_list = js['abstract']
    reference_summary_str = ' '.join(reference_summary_sent_list)
    reference_word_list = reference_summary_str.split(' ')
    reference_summary_len = len(reference_word_list)
    # find len bin
    reference_len_bin = LEN_BINS[reference_summary_len]
    reference_lower_len, reference_upper_len = LEN_BINS_RANGE[reference_len_bin]
    pred_len_bin = LEN_BINS[num_tokens]
    ref_len_bin_distance = abs(pred_len_bin - reference_len_bin)
    len_bin_square_difference_with_reference = (pred_len_bin - reference_len_bin) ** 2
    if reference_lower_len < num_tokens <= reference_upper_len:
        in_reference_bin = 1
    else:
        in_reference_bin = 0
    if target_len_bin >= 0:
        target_lower_len, target_upper_len = LEN_BINS_RANGE[target_len_bin]
        if target_lower_len < num_tokens <= target_upper_len:
            in_target_bin = 1
        else:
            in_target_bin = 0
        target_len_bin_distance = abs(pred_len_bin - target_len_bin)
        len_bin_square_difference_with_target = (pred_len_bin - target_len_bin) ** 2
    #variance
    len_square_difference_with_reference = (num_tokens - reference_summary_len)**2
    if num_tokens == reference_summary_len:
        match_reference_len = 1
    else:
        match_reference_len = 1
    if target_len >= 0:
        len_square_difference_with_target = (num_tokens - target_len) ** 2
        if num_tokens == target_len:
            match_target_len = 1
        else:
            match_target_len = 0

    doc_one_gram_counter.update(_make_n_gram(doc_word_list, n=1))
    doc_two_gram_counter.update(_make_n_gram(doc_word_list, n=2))
    doc_three_gram_counter.update(_make_n_gram(doc_word_list, n=3))

    if len(word_list_concated) > 0:
        if len(word_list_concated) >= 2 and word_list_concated[-2] in PREPOSITION:
            end_with_preposition = 1
        else:
            end_with_preposition = 0

        one_gram_repeat_fraction = n_gram_repeat(sum_one_gram_counter)
        #unique_one_gram_novelty, one_gram_novelty = n_gram_novelty(sum_one_gram_counter, doc_one_gram_counter)

        if len(word_list_concated) > 1:
            two_gram_repeat_fraction = n_gram_repeat(sum_two_gram_counter)
            #unique_two_gram_novelty, two_gram_novelty = n_gram_novelty(sum_two_gram_counter, doc_two_gram_counter)
        else:
            two_gram_repeat_fraction = 0
            #unique_two_gram_novelty = 0
            #two_gram_novelty = 0

        if len(word_list_concated) > 2:
            three_gram_repeat_fraction = n_gram_repeat(sum_three_gram_counter)
            #unique_three_gram_novelty, three_gram_novelty = n_gram_novelty(sum_three_gram_counter, doc_three_gram_counter)
        else:
            three_gram_repeat_fraction = 0
            #unique_three_gram_novelty = 0
            #three_gram_novelty = 0

    else:
        end_with_preposition = 0
        one_gram_repeat_fraction = 0
        two_gram_repeat_fraction = 0
        three_gram_repeat_fraction = 0
        #unique_one_gram_novelty = 0
        #unique_two_gram_novelty = 0
        #unique_three_gram_novelty = 0
        #one_gram_novelty = 0
        #two_gram_novelty = 0
        #three_gram_novelty = 0

    stat = {}
    stat['num_sentences'] = num_sentences
    stat['num_tokens'] = num_tokens
    stat['one_gram_repeat'] = one_gram_repeat_fraction
    stat['two_gram_repeat'] = two_gram_repeat_fraction
    stat['three_gram_repeat'] = three_gram_repeat_fraction
    stat['end_with_preposition'] = end_with_preposition
    stat['in_reference_bin'] = in_reference_bin
    stat['ref_len_bin_distance'] = ref_len_bin_distance
    if target_len_bin >= 0:
        stat['in_target_bin'] = in_target_bin
        stat['target_len_bin_distance'] = target_len_bin_distance
        stat['len_bin_square_difference_with_target'] = len_bin_square_difference_with_target
    stat['len_square_difference_with_reference'] = len_square_difference_with_reference
    stat['len_bin_square_difference_with_reference'] = len_bin_square_difference_with_reference
    stat['match_reference_len'] = match_reference_len
    if target_len >= 0:
        stat['len_square_difference_with_target'] = len_square_difference_with_target
        stat['match_target_len'] = match_target_len

    return stat


def main(args):
    output_path = join(args.decode_dir, "output")
    n_output = _count_data(output_path)
    total_num_sentences = 0
    total_num_tokens = 0
    one_gram_repeat_sum = 0
    two_gram_repeat_sum = 0
    three_gram_repeat_sum = 0
    num_tokens_list = []
    end_with_preposition_sum = 0
    total_in_reference_bin = 0
    total_in_target_bin = 0
    total_len_square_difference_with_reference = 0
    total_len_bin_square_difference_with_reference = 0
    total_len_square_difference_with_target = 0
    total_len_bin_square_difference_with_target = 0
    split_dir = join(args.data_dir, args.split)
    total_ref_len_bin_distance = 0
    total_target_len_bin_distance = 0
    total_match_reference_len = 0
    total_match_target_len = 0
    in_target_len_bin_all = []
    in_ref_len_bin_all = []

    for i in range(n_output):  # iterate every .dec
        dec_file_path = join(output_path, "{}.dec".format(i))
        data_file_path = join(split_dir, "{}.json".format(i))
        stat = count_lines_and_tokens(dec_file_path, data_file_path, args.target_len, args.target_len_bin, args.multi_ref)
        total_num_sentences += stat['num_sentences']
        total_num_tokens += stat['num_tokens']
        if total_num_tokens == 0:
            print("{} is empty!".format(i))
        one_gram_repeat_sum += stat['one_gram_repeat']
        two_gram_repeat_sum += stat['two_gram_repeat']
        three_gram_repeat_sum += stat['three_gram_repeat']
        num_tokens_list.append(stat['num_tokens'])
        end_with_preposition_sum += stat['end_with_preposition']
        total_in_reference_bin += stat['in_reference_bin']
        in_ref_len_bin_all.append(stat['in_reference_bin'])
        total_ref_len_bin_distance += stat['ref_len_bin_distance']
        total_len_square_difference_with_reference += stat['len_square_difference_with_reference']
        total_len_bin_square_difference_with_reference += stat['len_bin_square_difference_with_reference']
        total_match_reference_len += stat['match_reference_len']
        if args.target_len_bin >= 0:
            total_in_target_bin += stat['in_target_bin']
            in_target_len_bin_all.append(stat['in_target_bin'])
            total_target_len_bin_distance += stat['target_len_bin_distance']
            total_len_bin_square_difference_with_target += stat['len_bin_square_difference_with_target']
        if args.target_len >= 0:
            total_len_square_difference_with_target += stat['len_square_difference_with_target']
            total_match_target_len += stat['match_target_len']

    #print("average generated sentences: {:.3f}".format(total_num_sentences/n_output))
    #print("average number of tokens per summary: {:.3f}".format(total_num_tokens / n_output))
    #print("average tokens per sentence: {:.3f}".format(total_num_tokens/total_num_sentences))
    #print("average repeat 1-gram: {:.4f}".format(one_gram_repeat_sum / n_output))
    #print("average repeat 2-gram: {:.4f}".format(two_gram_repeat_sum / n_output))
    #print("average repeat 3-gram: {:.4f}".format(three_gram_repeat_sum / n_output))
    #print("average in reference bin: {:.4f}".format(total_in_reference_bin / n_output))
    #print("average reference len bin distance: {:.4f}".format(total_ref_len_bin_distance / n_output))
    #print("average len bin distance: {:.4f}".format(total_ref_len_bin_distance / n_output))
    #print("variance of len bin against the reference len bin: {:.4f}".format(total_len_bin_square_difference_with_reference / n_output * 0.001))
    #if args.target_len_bin >= 0:
    #    print("average in target bin: {:.4f}".format(total_in_target_bin / n_output))
    #    print("average target len bin distance: {:.4f}".format(total_target_len_bin_distance / n_output))
    #    print("variance of len bin against the target len bin: {:.4f}".format(
    #        total_len_bin_square_difference_with_target / n_output * 0.001))
    #print("variance of summary lengths against the reference length: {:.4f}".format(total_len_square_difference_with_reference / n_output * 0.001))
    #print("match reference len: {:.4f}".format(total_match_reference_len / n_output))
    if args.target_len >= 0:
        #print("variance of summary lengths against the desired length: {:.4f}".format(
        #    total_len_square_difference_with_target / n_output * 0.001))
        print("bin % of target len: {:.4f}".format(total_match_target_len / n_output))
        with open(join(args.decode_dir, 'target_len_bin.txt'), 'w') as f:
            for in_target_len in in_target_len_bin_all:
                f.write("{}\n".format(in_target_len))
    else:
        print("bin % of reference len: {:.4f}".format(total_in_reference_bin / n_output))
        with open(join(args.decode_dir, 'ref_len_bin.txt'), 'w') as f:
            for in_ref_len in in_ref_len_bin_all:
                f.write("{}\n".format(in_ref_len))
    #num_tokens_array = np.array(num_tokens_list)
    #print("min tokens: {}".format(num_tokens_array.min()))
    #print("max tokens: {}".format(num_tokens_array.max()))
    #print("std of tokens: {}".format(np.std(num_tokens_array)))
    #print("end with preposition: {}".format(end_with_preposition_sum))
    #with open(join(args.decode_dir, 'ref_len_bin.txt'), 'w') as f:
    #    for in_ref_len in in_ref_len_bin_all:
    #        f.write("{}\n".format(in_ref_len))


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
    parser.add_argument('-target_len', type=int, default=-1,
                        help='target length of summary (if any)')
    parser.add_argument('-target_len_bin', type=int, default=-1,
                        help='target length bin of summary (if any)')
    parser.add_argument('-multi_ref', action='store_true',
                        help='use if the test set has multiple reference summaries')

    args = parser.parse_args()
    main(args)
