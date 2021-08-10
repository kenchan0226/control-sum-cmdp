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
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import random
from metric import compute_rouge_l
from transformers import BertTokenizer


PS = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)
datetime_numerical_entity_types = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


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


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def stem_and_remove_stop_words(word_list):
    new_word_list = []
    for word in word_list:
        word = word.lower()
        if word not in STOP_WORDS:
            new_word_list.append(PS.stem(word))
    return new_word_list


def check_overlap(str_a, str_b):
    word_list_a = str_a.split(' ')
    word_list_b = str_b.split(' ')
    word_list_a_normalized = stem_and_remove_stop_words(word_list_a)
    word_list_b_normalized = stem_and_remove_stop_words(word_list_b)
    for word_a in word_list_a_normalized:
        for word_b in word_list_b_normalized:
            if word_a == word_b:
                return True
    return False


def check_exist(choice_list, entity_str):
    is_exist = False
    for choice in choice_list:
        if check_overlap(choice, entity_str):
            is_exist = True
            break
    return is_exist


def check_tuple_exists(entity_tuple_list, new_entity_tuple):
    is_exist = False
    new_entity_str, _ = new_entity_tuple
    for entity_tuple in entity_tuple_list:
        entity_str, _ = entity_tuple
        if check_overlap(entity_str, new_entity_str):
            is_exist = True
            break
    return is_exist


def extract_entities(sent_list):
    entity_list = []
    for sent in sent_list:
        for sent_spacy in nlp(sent).sents:
            for ent in sent_spacy.ents:
                entity_list.append((ent.text, ent.label_))
    return entity_list


def create_masked_question(ent):
    id_start = ent.start_char - ent.sent.start_char
    id_end = ent.end_char - ent.sent.start_char
    masked_question = ent.sent.text[:id_start] + \
                      "[MASK]" + ent.sent.text[id_end:]
    return masked_question

def select_ent_on_shortest_sentence(substitute_cand_list):
    shortest_len = 100000
    shortest_cand = None
    for cand in substitute_cand_list:
        cand_sent_len = len(cand.sent)
        if cand_sent_len < shortest_len:
            shortest_len = cand_sent_len
            shortest_cand = cand
    return shortest_cand


@curry
def process(data_dir, out_dir, json_i):
    #try:
    with open(join(data_dir, '{}.json'.format(json_i))) as f:
        js = json.loads(f.read())

    #if True:
    doc_sent_list = js['article']
    summary_sent_list = js['abstract']

    masked_question_list = []
    answer_list = []
    answer_entity_tuple_list = []
    num_doc_sents = len(doc_sent_list)
    first_400_non_datetime_numerical_question_indices = []
    #processed_entities = []
    entity_obj_list = []
    threshold = 0.6
    #negative_masked_question_list_same_ent_label = []
    #negative_context_list_same_ent_label = []
    ##negative_masked_question_list_diff_ent_label = []
    negative_context_list_diff_ent_label = []

    entities_count = 0
    if doc_sent_list and summary_sent_list:
        doc_word_list_lower = ' '.join(doc_sent_list).lower().split(' ')
        summary_word_list = ' '.join(summary_sent_list).split(' ')
        # print(summary_sent_list)
        summary_str = ' '.join(summary_word_list[:100])
        summary_spacy = nlp(summary_str)
        # iterate each entity
        for ent in summary_spacy.ents:
            #if ent.text in processed_entities:
            #    continue
            #processed_entities.append(ent.text)
            # skip datetime numerical entities
            if ent.label_ in datetime_numerical_entity_types:
                continue
            masked_question = create_masked_question(ent)
            #id_start = ent.start_char - ent.sent.start_char
            #id_end = ent.end_char - ent.sent.start_char
            #masked_question = ent.sent.text[:id_start] + \
            #                  "[MASK]" + ent.sent.text[id_end:]
            if len(masked_question.split(" ")) <= 2:
                continue
            entity_obj_list.append(ent)
            entities_count += 1
            masked_question_list.append(masked_question)
            answer_list.append(ent.text)
            answer_entity_tuple_list.append((ent.text, ent.label_))
            # print(ent.text)
            # print(masked_question)

            # check if present in first 400 words
            entity_start_end_list = check_present_named_entities(doc_word_list_lower,
                                                                 [ent.text.lower().split(' ')])
            # print(entity_start_end_list)
            entity_start, entity_end = entity_start_end_list[0]
            if 0 <= entity_end < 400:
                first_400_non_datetime_numerical_question_indices.append(len(masked_question_list) - 1)

        num_entities = len(entity_obj_list)
        # enumerate every pair of entities

        doc_sent_ent_obj_2dlist = []  # one entity list for each doc sent
        top10_doc_sent_list = doc_sent_list[:12]
        for doc_sent in top10_doc_sent_list:
            doc_sent_spacy = nlp(doc_sent)
            doc_sent_ent_obj_2dlist.append(doc_sent_spacy.ents)

        negative_masked_question_list = []
        negative_context_list = []
        # for each ref entity
        for ref_entity_obj in entity_obj_list:
            substitute_cand_list = []
            masked_question = create_masked_question(ref_entity_obj)
            # for each doc sent and its ent obj list
            for doc_sent, doc_sent_ent_obj_list in zip(top10_doc_sent_list, doc_sent_ent_obj_2dlist):
                # if ref entity not in doc sent
                # if ref_entity_obj.text not in doc_sent:
                if ref_entity_obj.text not in doc_sent and "-RRB-" not in doc_sent and \
                        compute_rouge_l(doc_sent.split(" "), ref_entity_obj.sent.text.split(" "), 'r') <= 0.2:
                    # for each entity in doc sent
                    for doc_sent_ent_obj in doc_sent_ent_obj_list:
                        # if non overlap and has the same entity type
                        if not check_overlap(ref_entity_obj.text, doc_sent_ent_obj.text) and \
                                ref_entity_obj.label_ == doc_sent_ent_obj.label_:
                            # add to candidate
                            substitute_cand_list.append(doc_sent_ent_obj)
            if len(substitute_cand_list) > 0:
                # randomly sample a candidate
                #random.shuffle(substitute_cand_list)
                #substitute_entity_obj = substitute_cand_list[0]
                substitute_entity_obj = select_ent_on_shortest_sentence(substitute_cand_list)
                # construct substitute sentence and insert it to summary
                substitute_sentence = substitute_entity_obj.sent.text
                substitute_sentence = substitute_sentence[:substitute_entity_obj.start_char - substitute_entity_obj.sent.start_char] + ref_entity_obj.text + substitute_sentence[substitute_entity_obj.end_char-substitute_entity_obj.sent.start_char:]
                context = summary_str[:ref_entity_obj.sent.start_char] + substitute_sentence + summary_str[ref_entity_obj.sent.end_char:]
                negative_context_list.append(context)

                # construct masked question
                #masked_question = create_masked_question(ref_entity_obj)
                negative_masked_question_list.append(masked_question)

            # if the sentence that contains ref_entity_obj has two entities, substitute another entity with ref_entity_obj
            if len(ref_entity_obj.sent.ents) == 2:
                for trg_ent in ref_entity_obj.sent.ents:
                    if not check_overlap(ref_entity_obj.text, trg_ent.text):
                        # if the entity has no overlap with ref_entity_obj, then substitute
                        sentence_with_repeated_entities = ref_entity_obj.sent.text[:trg_ent.start_char - ref_entity_obj.sent.start_char] + ref_entity_obj.text + ref_entity_obj.sent.text[trg_ent.end_char-ref_entity_obj.sent.start_char:]
                        # append negative question and context
                        negative_context_list.append(sentence_with_repeated_entities)
                        #masked_question = create_masked_question(ref_entity_obj)
                        negative_masked_question_list.append(masked_question)
                        #print("json {}".format(json_i))

        # when we cannot find entity
        if entities_count == 0:
            #print("json_{} does not have any entities".format(json_i))
            cloze_js = {"masked_question_list": []}
        else:

            cloze_js = {"masked_question_list": masked_question_list, "reference_summary_sent_list": summary_sent_list,
                        "answer_list": answer_list,
                        "first_400_non_datetime_numerical_question_indices": first_400_non_datetime_numerical_question_indices,
                        "negative_masked_question_list": negative_masked_question_list,
                        "negative_context_list": negative_context_list
                        }

            # prepare paraphrase questions
            scores = js["score"]
            ext_sent_ids = js["extracted"]
            salient_doc_sents = [doc_sent_list[i] for i in ext_sent_ids]
            salient_doc_str = " ".join(salient_doc_sents)
            all_larger_flag = True
            for score in scores:
                if score < threshold:
                    all_larger_flag = False
                    break

            all_entity_present_flag = True
            for ent_obj in entity_obj_list:
                if ent_obj.text not in salient_doc_str:
                    all_entity_present_flag = False
                    break

            salient_doc_str_tokenized = tokenizer.tokenize(salient_doc_str)
            if len(salient_doc_str_tokenized) <= 350:
                short_flag = True
            else:
                short_flag = False

            if all_larger_flag and all_entity_present_flag and short_flag:
                cloze_js["paraphrase_context"] = salient_doc_str
                #print("json {}".format(json_i))
            else:
                cloze_js["paraphrase_context"] = ""

    else:
        cloze_js = {"masked_question_list": []}

    with open(join(out_dir, '{}.json'.format(json_i)), 'w') as f:
        json.dump(cloze_js, f, indent=4)

    #except:
    #    print("json {} failed".format(json_i))
    #    cloze_js = {"masked_question_list": []}
    #    with open(join(out_dir, '{}.json'.format(json_i)), 'w') as f:
    #        json.dump(cloze_js, f, indent=4)


def label_mp(data, split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(data, split)
    out_dir = join(data, 'cloze_entity_squad_with_idx_and_unanswerable_and_paraphrase_and_repeat', split)
    os.makedirs(out_dir)
    n_data = _count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(data_dir, out_dir),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def label(data, split):
    """ process the data split with multi-processing"""
    data_dir = join(data, split)
    out_dir = join(data, 'cloze_entity_squad_with_idx_and_unanswerable_and_paraphrase_and_repeat', split)
    os.makedirs(out_dir)
    n_data = _count_data(data_dir)
    #n_data = 5
    for json_i in range(n_data):
        process(data_dir, out_dir, json_i)


def main(data, split):
    if split == 'all':
        for split in ['val', 'train', 'test']:
            label_mp(data, split)
            #label(data, split)
    else:
        label_mp(data, split)
        #label(data, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('')
    )
    parser.add_argument('-data', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    parser.add_argument('-seed', type=int, default=9527,
                        help='Random seed.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args.data, args.split)

