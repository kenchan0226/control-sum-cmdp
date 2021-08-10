import os
import json
from os.path import join
import re
import argparse
from extract_doc_sents_entities import find_entity_first_appearance_sentence_idx
import numpy as np
import pickle as pkl


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

    ref_entities_doc_sent_distribution = np.array([0] * 20)
    ref_entities_rank_distribution = np.array([0] * 20)

    for i in range(n_data):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))
        doc_sent_list = js['article']
        """
        doc_sent_ent_cloze_js = json.load(open(join(data_dir, 'doc_sents_entities_cloze', 'test', '{}.json'.format(i))))
        if len(doc_sent_ent_cloze_js["entities_masked_question_list_in_each_sent"]) == 0:
            continue
        """
        freq_ent_cloze_js = json.load(open(join(data_dir, 'doc_entities_frequencies_cloze', split, '{}.json'.format(i))))
        if len(freq_ent_cloze_js["masked_question_list_for_each_entity"]) == 0:
            continue
        entity_frequency_list = freq_ent_cloze_js['entity_frequency_list']

        reference_entity_list_non_numerical = js["reference_entity_list_non_numerical"]
        if len(reference_entity_list_non_numerical) == 0:
            continue
        for ref_entity in reference_entity_list_non_numerical:
            doc_sent_i = find_entity_first_appearance_sentence_idx(ref_entity, doc_sent_list)
            if doc_sent_i == -1:
                continue
            if doc_sent_i > 19:
                doc_sent_i = 19
            ref_entities_doc_sent_distribution[doc_sent_i] += 1

            #print(ref_entity)
            #print(doc_sent_i)

            for rank_i, ent_freq_tuple in enumerate(entity_frequency_list):
                #print(ent_freq_tuple)
                if ref_entity == ent_freq_tuple[0]:
                    if rank_i > 19:
                        rank_i = 19
                    ref_entities_rank_distribution[rank_i] += 1
                    #print(rank_i)
                    #print(ent_freq_tuple)

    ref_entities_doc_sent_distribution_normalized = ref_entities_doc_sent_distribution / ref_entities_doc_sent_distribution.sum()
    ref_entities_rank_distribution_normalized = ref_entities_rank_distribution / ref_entities_rank_distribution.sum()

    print(ref_entities_doc_sent_distribution_normalized)
    print(ref_entities_rank_distribution_normalized)

    #exit()
    with open(join(data_dir, 'ref_entities_doc_sent_distribution_{}.pkl'.format(split)), 'wb') as f:
        pkl.dump(ref_entities_doc_sent_distribution_normalized, f, pkl.HIGHEST_PROTOCOL)

    with open(join(data_dir, 'ref_entities_rank_distribution_{}.pkl'.format(split)), 'wb') as f:
        pkl.dump(ref_entities_rank_distribution_normalized, f, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('')
    )
    parser.add_argument('-data_dir', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store',
                        help='train or val or test.')

    args = parser.parse_args()

    main(args.data_dir, args.split)
