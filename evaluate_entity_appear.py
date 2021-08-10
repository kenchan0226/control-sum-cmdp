import re
from os.path import join
import json
import os
import argparse
import time
import numpy as np


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def _read_file_lower(filename):
    # print(dec_fname)
    summary_sent_list_lower = []
    with open(filename) as f:
        for _, l in enumerate(f):
            summary_sent_list_lower.append(l.strip().lower())
    summary_str_lower = ' '.join(summary_sent_list_lower)
    return summary_str_lower


def main(decode_dir, data):
    start_time = time.time()
    dec_dir = join(decode_dir, 'output')
    with open(join(decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    split_dir = join(data, split)
    ref_entity_appeared_ratio_all = []
    n_data = _count_data(split_dir)
    #n_data = 5
    for i in range(n_data):
        with open(join(split_dir, '{}.json'.format(i))) as f:
            js = json.loads(f.read())
        reference_entities = js['reference_entity_list_non_numerical']
        num_reference_entities = len(reference_entities)
        if num_reference_entities == 0:
            continue
        dec_fname = join(dec_dir, '{}.dec'.format(i))
        output_summary_str_lower = _read_file_lower(dec_fname)

        #print(reference_entities)
        #print(output_summary_str_lower)
        #print(num_reference_entities)
        num_appeared_entities = 0
        for entity_str in reference_entities:
            if entity_str.lower() in output_summary_str_lower:
                num_appeared_entities += 1
        ref_entity_appeared_ratio = num_appeared_entities / len(reference_entities)
        ref_entity_appeared_ratio_all.append(ref_entity_appeared_ratio)
        #print(num_appeared_entities)
        #print(ref_entity_appeared_ratio)
        #print()

    #print(avg_score.size())
    ref_entity_appeared_ratio_all = np.array(ref_entity_appeared_ratio_all)
    avg_ref_entity_appeared_ratio = ref_entity_appeared_ratio_all.mean()
    print("Average reference entities appeared ratio: {:.5}".format(avg_ref_entity_appeared_ratio))
    print("Processing time: {}s".format(time.time()-start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Evaluate Summa_qa')
    )
    parser.add_argument('--decode_dir', action='store', required=True, help='directory of decoded summaries')
    parser.add_argument('--data', action='store', required=True, help='directory of decoded summaries')
    args = parser.parse_args()
    main(args.decode_dir, args.data)
