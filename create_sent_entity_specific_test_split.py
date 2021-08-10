import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp
from cytoolz import concat, curry, compose
import argparse
import re
import spacy
import neuralcoref
import numpy as np
import random

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

MAX_OUT_LEN = 100

def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def check_present_named_entities(doc_word_list, named_entity_words_list):
    entity_start_end_list = []
    for entity_words in named_entity_words_list:  # for each named entity
        # check if it appears in document
        match = False
        for doc_start_idx in range(len(doc_word_list) - len(entity_words) + 1):
            match = True
            for entity_word_idx, entity_word in enumerate(entity_words):
                doc_word = doc_word_list[doc_start_idx + entity_word_idx]
                if doc_word != entity_word:
                    match = False
                    break
            if match:
                break
        if match:
            entity_start_end_list.append((doc_start_idx, doc_start_idx + len(entity_words)))
        else:
            entity_start_end_list.append((-1, -1))
    return entity_start_end_list


@curry
def process(in_data_dir, out_data_dir, cloze_data_dir, out_cloze_dir, portion, json_i):
    with open(join(in_data_dir, '{}.json'.format(json_i))) as f:
        js = json.loads(f.read())

    with open(join(cloze_data_dir, '{}.json'.format(json_i))) as f:
        cloze_js = json.loads(f.read())

    doc_sent_list = js['article']
    summary_sent_list = js['abstract']
    entities_masked_question_list_in_each_sent = cloze_js["entities_masked_question_list_in_each_sent"]

    if doc_sent_list and summary_sent_list and entities_masked_question_list_in_each_sent:
        entities_in_each_sent = cloze_js["entities_in_each_sent"]
        answer_list_in_each_sent = cloze_js["answer_list_in_each_sent"]
        # truncate summary_sent_list
        summary_sent_list_trunc = []
        #for summary_sent in summary_sent_list:
        #    summary_sent_word_list_trunc = summary_sent.split(' ')[:MAX_OUT_LEN]
        #    summary_sent_trunc = ' '.join(summary_sent_word_list_trunc)
        #    summary_sent_list_trunc.append(summary_sent_trunc)
        """
        if portion == 0:
            low = 0
            high = 3
        elif portion == 1:
            low = 3
            high = 6
        elif portion == 2:
            low = 6
            high = 9
        """
        if portion == 0:
            low = 0
            high = 2
        elif portion == 1:
            low = 2
            high = 4
        elif portion == 2:
            low = 4
            high = 6
        elif portion == 3:
            low = 6
            high = 8
        else:
            raise ValueError

        preselected_entities_list = []
        all_entities_masked_question_list = []
        all_answer_list = []

        for i in range(low, high):
            preselected_entities_list += entities_in_each_sent[i]

        for i in range(len(entities_masked_question_list_in_each_sent)):
            all_entities_masked_question_list += entities_masked_question_list_in_each_sent[i]
            all_answer_list += answer_list_in_each_sent[i]

        #print(entities_in_each_sent[:3])

        # randomly pick three entities
        """
        if len(preselected_entities_list) > 5:
            #selected_indices = random.choices(list(range(len(preselected_entities_list))), k=3)
            #selected_entities_list = [preselected_entities_list[idx] for idx in selected_indices]
            #print("random draw")
            #print(selected_indices)
            selected_entities_list = preselected_entities_list[:5]
        else:
            selected_entities_list = preselected_entities_list
        """
        selected_entities_list = preselected_entities_list[:5]

        if len(selected_entities_list) > 0:
            selected_entities_list_str = ' <ent> '.join(selected_entities_list) + ' <ent_end>'
        else:
            selected_entities_list_str = ""

        #print(selected_entities_list)
        #print(selected_entities_list_str)

        # find start and end position for each entity
        selected_entities_words_list_lower = [entity_str.lower().split(' ') for entity_str in selected_entities_list]
        doc_word_list_lower = ' '.join(doc_sent_list).lower().split(' ')
        entity_start_end_list = check_present_named_entities(doc_word_list_lower, selected_entities_words_list_lower)

        selected_entities_list_filtered = []
        entity_start_end_list_filtered = []
        for entity_str, (entity_start, entity_end) in zip(selected_entities_list, entity_start_end_list):
            if entity_end >= 0:
                selected_entities_list_filtered.append(entity_str)
                entity_start_end_list_filtered.append((entity_start, entity_end))
        selected_entities_list = selected_entities_list_filtered
        entity_start_end_list = entity_start_end_list_filtered

        selected_entities_masked_question_list = []
        selected_answer_list = []

        # select masked questions
        for entities_masked_question, answer in zip(all_entities_masked_question_list, all_answer_list):
            if answer in selected_entities_list:
                selected_entities_masked_question_list.append(entities_masked_question)
                selected_answer_list.append(answer)

        #print(selected_entities_masked_question_list)
        #print(selected_multiple_choices_list)
        #print(selected_answer_list)
        #print(selected_answer_idx_list)
        #print()

        # write to new testing split
        js['reference_entity_list_non_numerical'] = selected_entities_list
        js['reference_entity_list_non_numerical_str'] = selected_entities_list_str
        js['reference_entity_start_end_list'] = entity_start_end_list
        js['masked_question_ids_list'] = []
        js['answer_list'] = []
        js["reference_coref_clusters"] = []

        with open(join(out_data_dir, '{}.json'.format(json_i)), 'w') as f:
            json.dump(js, f, indent=4)

        # write to cloze_entity_coref_with_idx
        out_cloze_js = {"masked_question_list": selected_entities_masked_question_list,
                        "answer_list": selected_answer_list
                        }
        with open(join(out_cloze_dir, '{}.json'.format(json_i)), 'w') as f:
            json.dump(out_cloze_js, f, indent=4)

    else:
        js['reference_entity_list_non_numerical'] = []
        js['reference_entity_start_end_list'] = []
        js['reference_entity_list_non_numerical_str'] = ""

        with open(join(out_data_dir, '{}.json'.format(json_i)), 'w') as f:
            json.dump(js, f, indent=4)

        out_cloze_js = {"masked_question_list": []}
        with open(join(out_cloze_dir, '{}.json'.format(json_i)), 'w') as f:
            json.dump(out_cloze_js, f, indent=4)


def label_mp(data, out_split, in_split, portion):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(in_split))
    in_data_dir = join(data, in_split)
    out_data_dir = join(data, out_split)
    cloze_data_dir = join(data, "doc_sents_entities_cloze", in_split)
    out_cloze_dir = join(data, 'cloze_entity_squad_with_idx_and_unanswerable_and_paraphrase_and_repeat', out_split)
    os.makedirs(out_data_dir)
    os.makedirs(out_cloze_dir)
    n_data = _count_data(in_data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(in_data_dir, out_data_dir, cloze_data_dir, out_cloze_dir, portion),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def label(data, out_split, in_split, portion):
    """ process the data split with multi-processing"""
    in_data_dir = join(data, in_split)
    out_data_dir = join(data, out_split)
    cloze_data_dir = join(data, "doc_sents_entities_cloze", in_split)
    out_cloze_dir = join(data, 'cloze_entity_squad_with_idx_and_unanswerable_and_paraphrase_and_repeat', out_split)
    os.makedirs(out_data_dir)
    os.makedirs(out_cloze_dir)
    n_data = _count_data(in_data_dir)
    #n_data = 5
    for i in range(n_data):
        process(in_data_dir, out_data_dir, cloze_data_dir, out_cloze_dir, portion, i)


def main(data, out_split, in_split, portion):
    if in_split == 'all':
        for split in ['val', 'train', 'test']:
            label_mp(data, out_split, in_split, portion)
    else:
        label_mp(data, out_split, in_split, portion)
        #label(data, out_split, in_split, portion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('')
    )
    parser.add_argument('-data', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-out_split', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-portion', type=int, action='store',
                        help='The directory of the data.')
    parser.add_argument('-in_split', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    parser.add_argument('-seed', type=int, action='store', default=9527,
                        help='The directory of the data.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args.data, args.out_split, args.in_split, args.portion)

