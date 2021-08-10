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

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

DATE_TIME_NUMERICAL_ENTITIES_TYPES = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
MAX_OUT_LEN = 100

def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def extract_entities(sent_list):
    entity_list = []
    num_entities_for_each_sent = [0] * len(sent_list)
    for sent_i, sent in enumerate(sent_list):
        for sent_spacy in nlp(sent).sents:
            for ent in sent_spacy.ents:
                if ent.label_ not in DATE_TIME_NUMERICAL_ENTITIES_TYPES:
                    num_entities_for_each_sent[sent_i] += 1
                    entity_list.append(ent.text)
    return entity_list, num_entities_for_each_sent


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
def process(in_data_dir, out_data_dir, i):
    #if True:
    try:
        with open(join(in_data_dir, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        #with open(join(out_data_dir, '{}.json'.format(i))) as f:
            #out_js = json.loads(f.read())

        doc_sent_list = js['article']
        summary_sent_list = js['abstract']

        # truncate summary_sent_list
        summary_sent_list_trunc = []
        #for summary_sent in summary_sent_list:
        #    summary_sent_word_list_trunc = summary_sent.split(' ')[:MAX_OUT_LEN]
        #    summary_sent_trunc = ' '.join(summary_sent_word_list_trunc)
        #    summary_sent_list_trunc.append(summary_sent_trunc)

        if doc_sent_list and summary_sent_list:
            doc_word_list_lower = ' '.join(doc_sent_list).lower().split(' ')
            # truncate summary up to 100 tokens
            summary_word_list = ' '.join(summary_sent_list).split(' ')
            #print(summary_sent_list)
            summary_str = ' '.join(summary_word_list[:100])
            #print(summary_str)
            # extract coref and named entity:
            summary_spacy = nlp(summary_str)
            coref_clusters = {}
            reference_entity_list_non_numerical = []
            processed_entities = []
            for ent in summary_spacy.ents:
                # check if it is non numerical
                if ent.label_ not in DATE_TIME_NUMERICAL_ENTITIES_TYPES and ent.text not in processed_entities:
                    processed_entities.append(ent.text)
                    coref = ent._.coref_cluster
                    if coref is not None:
                        coref_clusters[ent.text] = [mention.text for mention in coref.mentions]
                        reference_entity_list_non_numerical.append(ent.text)
                    else:
                        coref_clusters[ent.text] = [ent.text]
                        reference_entity_list_non_numerical.append(ent.text)
            # check if present in the first 400 words
            reference_named_entity_words_list_lower = [entity_str.lower().split(' ') for entity_str in reference_entity_list_non_numerical]
            #print(coref_clusters)
            #print(reference_named_entity_words_list_lower)
            entity_start_end_list = check_present_named_entities(doc_word_list_lower, reference_named_entity_words_list_lower)
            #print(entity_start_end_list)
            #for entity_start, entity_end in entity_start_end_list:
            #    print(doc_word_list_lower[entity_start: entity_end])
            # remove entities not exists or not in first 400 words, precompute position_ids
            filtered_reference_entity_list_non_numerical = []
            filtered_entity_start_end_list = []
            for entity_str, (entity_start, entity_end) in zip(reference_entity_list_non_numerical, entity_start_end_list):
                if 0 <= entity_end < 400:
                    filtered_reference_entity_list_non_numerical.append(entity_str)
                    filtered_entity_start_end_list.append((entity_start, entity_end))
                else:
                    coref_clusters.pop(entity_str, None)
            # remove summary sentences with no reference entities or its mentions
            if len(filtered_reference_entity_list_non_numerical) > 0:
                summary_sent_list_filtered = []
                for summary_sent_i, summary_sent in enumerate(summary_sent_list):
                    match_flag = False
                    for entity, mentions in coref_clusters.items():
                        for mention in mentions:
                            if mention in summary_sent:
                                match_flag = True
                                break
                        if match_flag:
                            break
                    if match_flag:
                        summary_sent_list_filtered.append(summary_sent)
            else:
                summary_sent_list_filtered = summary_sent_list

            #print(filtered_reference_entity_list_non_numerical)
            #print(filtered_entity_start_end_list)
            #print()

            js['reference_entity_list_non_numerical'] = filtered_reference_entity_list_non_numerical
            js['reference_entity_start_end_list'] = filtered_entity_start_end_list
            if len(filtered_reference_entity_list_non_numerical) > 0:
                js['reference_entity_list_non_numerical_str'] = ' <ent> '.join(filtered_reference_entity_list_non_numerical) + ' <ent_end>'
            else:
                js['reference_entity_list_non_numerical_str'] = ""
            js['reference_coref_clusters'] = coref_clusters
            js['abstract'] = summary_sent_list_filtered

            # remove some unused keys
            js.pop("extractive_fragment_density", None)
            js.pop("extractive_fragment_coverage", None)
            js.pop("similar_source_indices_lebanoff", None)
            js.pop("avg_fusion_ratio", None)
            js.pop("unique_two_gram_novelty", None)
            js.pop("two_gram_novelty", None)

            with open(join(out_data_dir, '{}.json'.format(i)), 'w') as f:
                json.dump(js, f, indent=4)

        else:

            js['reference_entity_list_non_numerical'] = []
            js['reference_entity_start_end_list'] = []
            js['reference_entity_list_non_numerical_str'] = ""

            with open(join(out_data_dir, '{}.json'.format(i)), 'w') as f:
                json.dump(js, f, indent=4)

    except:
        #print("json {} failed".format(i))
        pass

def label_mp(in_data, out_data, split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    in_data_dir = join(in_data, split)
    out_data_dir = join(out_data, split)
    os.makedirs(out_data_dir)
    n_data = _count_data(in_data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(in_data_dir, out_data_dir),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def label(in_data, out_data, split):
    """ process the data split with multi-processing"""
    in_data_dir = join(in_data, split)
    out_data_dir = join(out_data, split)
    os.makedirs(out_data_dir)
    n_data = _count_data(in_data_dir)
    #n_data = 16
    for i in range(n_data):
        process(in_data_dir, out_data_dir, i)


def main(in_data, out_data, split):
    if split == 'all':
        for split in ['val', 'train', 'test']:
            label_mp(in_data, out_data, split)
    else:
        label_mp(in_data, out_data, split)
        #label(in_data, out_data, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('')
    )
    parser.add_argument('-in_data', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-out_data', type=str, action='store',
                        help='The directory of the data.')

    parser.add_argument('-split', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    args = parser.parse_args()
    main(args.in_data, args.out_data, args.split)

