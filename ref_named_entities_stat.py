import os
import json
from os.path import join
import re
import argparse
import numpy as np
from utils.io import ext_frag_density_to_bin, fusion_ratio_to_bin


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
    total_num_ref_entities = 0
    total_num_no_entities = 0
    total_num_duplicate_entities = 0

    for i in range(n_data):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))
        if js['article'] and js['abstract']:
            total_num_samples += 1
            doc_sent_list = js['article']
            num_doc_sents = len(doc_sent_list)
            doc_str = ' '.join(doc_sent_list).lower()
            doc_word_list = doc_str.split(' ')
            num_doc_tokens = len(doc_word_list)
            ref_entities = js['reference_entity_list_non_numerical']
            processed_entities = []
            duplicate_entities = 0
            for entity in ref_entities:
                if entity in processed_entities:
                    duplicate_entities += 1
                processed_entities.append(entity)
            num_ref_entities = len(ref_entities)
            total_num_ref_entities += num_ref_entities
            if num_ref_entities == 0:
                total_num_no_entities += 1
            total_num_duplicate_entities += duplicate_entities

    avg_summary_tokens = total_num_summary_tokens/total_num_samples

    print("avg. no. of reference entities:\t{:.3f}".format(total_num_ref_entities/total_num_samples))
    print("no. of samples with no entities:\t{}".format(total_num_no_entities))
    print("no. of duplicate entities:\t{}".format(total_num_duplicate_entities))

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
