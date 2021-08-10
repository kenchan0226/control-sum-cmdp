import numpy as np
from utils.string_helper import _make_n_gram
from collections import Counter
import torch
from utils.pretrained_discriminator import SeqClassifyDiscriminator
#from nltk.corpus import stopwords
import textstat
from utils.io import LEN_BINS_RANGE, LEN_BINS, n_gram_novelty_to_bin, ext_frag_density_to_bin, fusion_ratio_to_bin
import pickle as pkl
from nltk.corpus import stopwords
from dataset_extractive_fragment_stat import compute_extractive_fragment, compute_extractive_fragment_density
import ssi_functions
import multiprocessing as mp
import spacy
import neuralcoref
from utils.cloze_mc_model import ClozeMCModel
from utils.cloze_model import ClozeModel
import os
import json
#num_cpus = mp.cpu_count()
from utils.time_log import time_since
import time
import re

NUMERICAL_ENTITIES_TYPES = ["PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]


def count_named_entity_appear_frequency(doc_word_list, entity_words):
    # check if it appears in document
    match = False
    appear_frequency = 0
    for doc_start_idx in range(len(doc_word_list) - len(entity_words) + 1):
        match = True
        for entity_word_idx, entity_word in enumerate(entity_words):
            doc_word = doc_word_list[doc_start_idx + entity_word_idx]
            if doc_word.lower() != entity_word.lower():
                match = False
                break
        if match:
            appear_frequency += 1
    return appear_frequency


def compute_n_gram_novelty(pred_word_list, src_word_list, n):
    pred_n_gram_counter = Counter()
    pred_n_gram_counter.update(_make_n_gram(pred_word_list, n))

    src_n_gram_counter = Counter()
    src_n_gram_counter.update(_make_n_gram(src_word_list, n))

    num_pred_n_grams = sum(pred_n_gram_counter.values())
    num_novel_pred_n_grams = 0
    for n_gram, cnt in pred_n_gram_counter.items():
        if n_gram not in src_n_gram_counter:
            num_novel_pred_n_grams += cnt

    novel_n_gram_fraction = num_novel_pred_n_grams / num_pred_n_grams
    return novel_n_gram_fraction


class StyleDiscriminatorCost:
    def __init__(self, device):
        self.device = device
        model_dir = "saved_model/bert_classifier_xsum_fox_weighted_sampler.bert_classifier.20191021-211528"
        ckpt_dir = model_dir + "/ckpt/epoch-3-total_batch-20000-valid_f1-0.9635"
        self.discriminator_model = SeqClassifyDiscriminator(model_dir, ckpt_dir, device=device)

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        pred_str_list = [' '.join(pred_word_list) for pred_word_list in pred_word_2d_list]
        class_prob = self.discriminator_model.score(pred_str_list)  # [batch, 2]
        return class_prob[:, 1]  # [batch]


class HighReadabilityCosts:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        batch_cost = []
        for pred_word_list in pred_word_2d_list:
            pred_str = ' '.join(pred_word_list)
            flesch_reading_ease_score = textstat.flesch_reading_ease(pred_str)
            if flesch_reading_ease_score >= 45:
                cost = 1.0
            else:
                cost = 0.0
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class LowReadabilityCosts:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        batch_cost = []
        for pred_word_list in pred_word_2d_list:
            pred_str = ' '.join(pred_word_list)
            flesch_reading_ease_score = textstat.flesch_reading_ease(pred_str)
            if flesch_reading_ease_score < 55:
                cost = 1.0
            else:
                cost = 0.0
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class LengthBinConsistent:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        len_bin_list = control_variables['len_bins']
        batch_cost = []
        for pred_word_list, len_bin in zip(pred_word_2d_list, len_bin_list):
            pred_len = len(pred_word_list)
            lower_len, upper_len = LEN_BINS_RANGE[len_bin]
            if lower_len < pred_len <= upper_len:
                cost = 0.0
            else:
                cost = 1.0
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class LengthBinDistance:
    def __init__(self, device, total_len_bins=10):
        self.device = device
        self.total_len_bins = 10

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        target_len_bin_list = control_variables['len_bins']
        batch_cost = []
        for pred_word_list, target_len_bin in zip(pred_word_2d_list, target_len_bin_list):
            pred_len = len(pred_word_list)
            pred_len_bin = LEN_BINS[pred_len]
            len_bin_distance = abs(target_len_bin - pred_len_bin) / self.total_len_bins
            batch_cost.append(len_bin_distance)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class LengthBinDistanceUnnormalized:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        target_len_bin_list = control_variables['len_bins']
        batch_cost = []
        for pred_word_list, target_len_bin in zip(pred_word_2d_list, target_len_bin_list):
            pred_len = len(pred_word_list)
            pred_len_bin = LEN_BINS[pred_len]
            len_bin_distance = abs(target_len_bin - pred_len_bin)
            batch_cost.append(len_bin_distance)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class ExactLengthCost:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        target_len_list = control_variables['exact_lens']
        batch_cost = []
        for pred_word_list, target_len in zip(pred_word_2d_list, target_len_list):
            pred_len = len(pred_word_list)
            """
            if pred_len == target_len:
                cost = 0.0
            else:
                cost = 1.0
            """
            cost = abs(pred_len - target_len)
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class ExactLengthCostDistance:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        target_len_list = control_variables['exact_lens']
        batch_cost = []
        for pred_word_list, target_len in zip(pred_word_2d_list, target_len_list):
            pred_len = len(pred_word_list)
            cost = abs(pred_len - target_len) / target_len
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class ExactLengthCostDistanceUnnormalized:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        target_len_list = control_variables['exact_lens']
        batch_cost = []
        for pred_word_list, target_len in zip(pred_word_2d_list, target_len_list):
            pred_len = len(pred_word_list)
            cost = abs(pred_len - target_len)
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class NegativeNamedEntityF1:
    def __init__(self, device):
        self.device = device
        self.nlp = spacy.load("en_core_web_sm")
        neuralcoref.add_to_pipe(self.nlp)
        self.beta = 2.0

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # reference entities_list and output entities_list should be lower-cased
        reference_entities_list = control_variables['reference_entities_list']
        batch_cost = []
        for pred_word_list, reference_entities in zip(pred_word_2d_list, reference_entities_list):
            num_matched_entities = 0
            num_reference_entities = len(reference_entities)
            unique_output_entities = []
            if num_reference_entities > 0:
                num_unique_output_entities = 0
                pred_str = ' '.join(pred_word_list)
                pred_str_spacy = self.nlp(pred_str)
                for ent in pred_str_spacy.ents:
                    if ent.label_ in NUMERICAL_ENTITIES_TYPES or ent.text in unique_output_entities:
                        continue
                    unique_output_entities.append(ent.text)
                    num_unique_output_entities += 1
                    if ent.text.lower() in reference_entities:
                        num_matched_entities += 1
                if num_unique_output_entities > 0:
                    precision = num_matched_entities / num_unique_output_entities
                    recall = num_matched_entities / num_reference_entities
                    if precision == 0 or recall == 0:
                        f_beta = 0.0
                    else:
                        f_beta = (1+self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall)
                        #f_beta = 2 * (precision * recall) / (precision + recall)
                else:
                    f_beta = 0.0
            else:
                f_beta = 1.0
            batch_cost.append(-f_beta)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class NegativeNamedEntityClozeConfidence:
    def __init__(self, device, threshold):
        self.device = device
        self.threshold = threshold
        self.cloze_model = ClozeMCModel(self.device)

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
              trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        masked_questions_ids_2dlist = control_variables['masked_questions_ids_2dlist']
        answer_id_2dlist = control_variables['answer_id_2dlist']
        multiple_choices_ids_2dlist = control_variables['multiple_choices_ids_2dlist']
        #print(masked_questions_ids_2dlist)
        #print(answer_id_2dlist)
        #print(multiple_choices_ids_2dlist)
        #print()
        summary_str_list = [' '.join(pred_word_list) for pred_word_list in pred_word_2d_list]
        # feed a batch to the model, record the position of each sample
        num_questions_per_sample = [len(questions) for questions in masked_questions_ids_2dlist]
        #print(num_questions_per_sample)
        flattened_masked_questions_ids_list = []
        flattened_answer_id_list = []
        flattened_multiple_choices_ids_list = []
        flattened_context_str_list = []
        for masked_question_ids_list, answer_id_list, multiple_choices_ids_list, summary_str in zip(
                masked_questions_ids_2dlist, answer_id_2dlist, multiple_choices_ids_2dlist, summary_str_list):
            flattened_masked_questions_ids_list += masked_question_ids_list
            flattened_answer_id_list += answer_id_list
            flattened_multiple_choices_ids_list += multiple_choices_ids_list
            flattened_context_str_list += [summary_str] * len(masked_question_ids_list)
            #print(summary_str)

        #print(flattened_context_str_list)
        #print(len(flattened_context_str_list))
        #print(len(flattened_masked_questions_ids_list))
        #print(len(flattened_answer_id_list))
        #print(len(flattened_multiple_choices_ids_list))
        #print()

        confidence_score = self.cloze_model.compute_confidence_score(flattened_masked_questions_ids_list,
                                                                flattened_multiple_choices_ids_list,
                                                                flattened_answer_id_list,
                                                                flattened_context_str_list)
        # confidence_score: [len(flattened_masked_questions_ids_list)]
        # compute average confidence for each sample
        num_processed_samples = 0
        score_for_each_batch = []
        #print(flattened_context_str_list)
        #print(confidence_score)
        for i in range(len(num_questions_per_sample)):
            # average for each batch
            if summary_str_list[i].strip() == "":
                score_for_each_batch.append(torch.FloatTensor([0.0]).to(self.device))
            elif num_questions_per_sample[i] > 0:
                avg_score = confidence_score[
                            num_processed_samples:num_processed_samples + num_questions_per_sample[i]].mean(dim=0)
                score_for_each_batch.append(avg_score)
            else:
                score_for_each_batch.append(torch.FloatTensor([self.threshold]).to(self.device))
            #print(num_processed_samples)
            #print(num_processed_samples + num_questions_per_sample[i])
            num_processed_samples += num_questions_per_sample[i]
        score_for_each_batch = torch.stack(score_for_each_batch, dim=0)
        # [batch_size]
        #print(-score_for_each_batch)
        #print(self.threshold)
        #print()
        #exit()
        return -score_for_each_batch

"""
class NegativeNamedEntityQAF1LengthNormalized:
    def __init__(self, device, threshold):
        self.device = device
        self.threshold = threshold
        self.cloze_model = ClozeModel(self.device)

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
              trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        masked_questions_ids_2dlist = control_variables['masked_questions_ids_2dlist']
        answer_str_2dlist = control_variables['answer_2dlist']
        #print(masked_questions_ids_2dlist)
        #print(answer_str_2dlist)
        #print()
        summary_str_list = [' '.join(pred_word_list) for pred_word_list in pred_word_2d_list]
        trg_summary_lens = [len(trg_word_list) for trg_word_list in trg_word_2d_list]

        # feed a batch to the model, record the position of each sample
        num_questions_per_sample = [len(questions) for questions in masked_questions_ids_2dlist]
        #print(num_questions_per_sample)
        flattened_masked_questions_ids_list = []
        flattened_answer_str_list = []
        flattened_context_str_list = []
        for masked_question_ids_list, answer_str_list, summary_str in zip(
                masked_questions_ids_2dlist, answer_str_2dlist, summary_str_list):
            flattened_masked_questions_ids_list += masked_question_ids_list
            flattened_answer_str_list += answer_str_list
            flattened_context_str_list += [summary_str] * len(masked_question_ids_list)
            #print(summary_str)

        #print(flattened_context_str_list)
        #print(len(flattened_context_str_list))
        #print(len(flattened_masked_questions_ids_list))
        #print(len(flattened_answer_str_list))
        #print()

        f1_score = self.cloze_model.compute_f1_score(flattened_masked_questions_ids_list,
                                                                flattened_answer_str_list,
                                                                flattened_context_str_list)
        # f1_score: [len(flattened_masked_questions_ids_list)]
        # compute average confidence for each sample
        num_processed_samples = 0
        score_for_each_batch = []
        #print(flattened_context_str_list)
        #print(confidence_score)
        for i in range(len(num_questions_per_sample)):
            # average for each batch
            if summary_str_list[i].strip() == "":
                score_for_each_batch.append(torch.tensor(0.0).to(self.device))
            elif num_questions_per_sample[i] > 0:
                avg_score = f1_score[
                            num_processed_samples:num_processed_samples + num_questions_per_sample[i]].mean(dim=0)
                score_for_each_batch.append(avg_score)
            else:
                score_for_each_batch.append(torch.tensor(-self.threshold).to(self.device))
            #print(num_processed_samples)
            #print(num_processed_samples + num_questions_per_sample[i])
            num_processed_samples += num_questions_per_sample[i]
        #print(score_for_each_batch)
        #print(score_for_each_batch[0].size())
        score_for_each_batch = torch.stack(score_for_each_batch, dim=0)
        # [batch_size]
        #print(score_for_each_batch.size())
        #print(score_for_each_batch)
        #print(self.threshold)
        #print()
        #exit()
        return -score_for_each_batch
"""

class NegativeNamedEntityQAF1:
    def __init__(self, device, threshold):
        self.device = device
        self.threshold = threshold
        self.cloze_model = ClozeModel(self.device)

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
              trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        masked_questions_ids_2dlist = control_variables['masked_questions_ids_2dlist']
        answer_str_2dlist = control_variables['answer_2dlist']
        #print(masked_questions_ids_2dlist)
        #print(answer_str_2dlist)
        #print()
        summary_str_list = [' '.join(pred_word_list) for pred_word_list in pred_word_2d_list]
        # feed a batch to the model, record the position of each sample
        num_questions_per_sample = [len(questions) for questions in masked_questions_ids_2dlist]
        #print(num_questions_per_sample)
        flattened_masked_questions_ids_list = []
        flattened_answer_str_list = []
        flattened_context_str_list = []
        for masked_question_ids_list, answer_str_list, summary_str in zip(
                masked_questions_ids_2dlist, answer_str_2dlist, summary_str_list):
            flattened_masked_questions_ids_list += masked_question_ids_list
            flattened_answer_str_list += answer_str_list
            flattened_context_str_list += [summary_str] * len(masked_question_ids_list)
            #print(summary_str)

        #print(flattened_context_str_list)
        #print(len(flattened_context_str_list))
        #print(len(flattened_masked_questions_ids_list))
        #print(len(flattened_answer_str_list))
        #print()

        f1_score = self.cloze_model.compute_f1_score(flattened_masked_questions_ids_list,
                                                                flattened_answer_str_list,
                                                                flattened_context_str_list)
        # f1_score: [len(flattened_masked_questions_ids_list)]
        # compute average confidence for each sample
        num_processed_samples = 0
        score_for_each_batch = []
        #print(flattened_context_str_list)
        #print(confidence_score)
        for i in range(len(num_questions_per_sample)):
            # average for each batch
            if summary_str_list[i].strip() == "":
                score_for_each_batch.append(torch.tensor(0.0).to(self.device))
            elif num_questions_per_sample[i] > 0:
                avg_score = f1_score[
                            num_processed_samples:num_processed_samples + num_questions_per_sample[i]].mean(dim=0)
                score_for_each_batch.append(avg_score)
            else:
                score_for_each_batch.append(torch.tensor(-self.threshold).to(self.device))
            #print(num_processed_samples)
            #print(num_processed_samples + num_questions_per_sample[i])
            num_processed_samples += num_questions_per_sample[i]
        #print(score_for_each_batch)
        #print(score_for_each_batch[0].size())
        score_for_each_batch = torch.stack(score_for_each_batch, dim=0)
        # [batch_size]
        #print(score_for_each_batch.size())
        #print(score_for_each_batch)
        #print(self.threshold)
        #print()
        #exit()
        return -score_for_each_batch


class EntityRepeatCost:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        batch_cost = []
        reference_entities_list = control_variables['reference_entities_list']
        for pred_word_list_sent_tokenized, reference_entities in zip(pred_word_2d_list_sent_tokenized, reference_entities_list):
            num_pred_sents = len(pred_word_list_sent_tokenized)
            num_sents_contain_repeat = 0
            #print("Sample")
            for pred_word_list in pred_word_list_sent_tokenized:
                #print("pred_str")
                #print(" ".join(pred_word_list))
                for reference_ent_str in reference_entities:
                    entity_word_list = reference_ent_str.split(" ")
                    #print("entity: {}".format(entity_word_list))
                    #print("pred word list: {}".format(pred_word_list))
                    cnt = count_named_entity_appear_frequency(pred_word_list, entity_word_list)
                    #cnt = pred_str.count(reference_ent_str)
                    #print("cnt: {}".format(cnt))
                    if cnt >= 2:
                        num_sents_contain_repeat += 1
                        break
            if num_pred_sents == 0:
                fraction_of_sents_contains_repeat = 0.0
            else:
                fraction_of_sents_contains_repeat = num_sents_contain_repeat / num_pred_sents
            #print("cost: {}".format(fraction_of_sents_contains_repeat))
            #print()
            batch_cost.append(fraction_of_sents_contains_repeat)
        #exit()
        return torch.FloatTensor(batch_cost).to(self.device)


class PPDBCost:
    def __init__(self, device, is_negative=False):
        self.device = device
        self.is_negative = is_negative
        self.phrase_dict_path = '/research/king3/hpchan/datasets/SimplePPDB/ppdb_phrase_to_paraphrase_count.pkl'
        with open(self.phrase_dict_path, 'rb') as f:
            self.phrase_dict = pkl.load(f)

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        batch_cost = []
        for pred_word_list in pred_word_2d_list:
            pred_three_gram_list = self._make_n_gram_phrase(pred_word_list, n=3)
            pred_two_gram_list = self._make_n_gram_phrase(pred_word_list, n=2)
            total_num_difficult_phrases = 0
            total_num_simple_phrases = 0
            total_num_draw_phrases = 0
            for phrase_list in [pred_three_gram_list, pred_two_gram_list, pred_word_list]:
                num_difficult_phrases, num_simple_phrases, num_draw_phrases = self.compute_difficulty(phrase_list, self.phrase_dict)
                total_num_difficult_phrases += num_difficult_phrases
                total_num_simple_phrases += num_simple_phrases
                total_num_draw_phrases += num_draw_phrases
            relative_difficult_phrase_frequency = total_num_difficult_phrases / (total_num_difficult_phrases + total_num_simple_phrases)
            if self.is_negative:
                relative_difficult_phrase_frequency = -relative_difficult_phrase_frequency
            batch_cost.append(relative_difficult_phrase_frequency)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]

    def compute_difficulty(self, phrase_list, phrase_dict):
        num_difficult_phrases = 0
        num_simple_phrases = 0
        num_draw_phrases = 0
        for phrase in phrase_list:
            if phrase in phrase_dict:
                num_simpler_paraphrases, num_more_difficulty_paraphrases = phrase_dict[phrase]
                if num_simpler_paraphrases > num_more_difficulty_paraphrases:
                    num_difficult_phrases += 1
                elif num_simpler_paraphrases < num_more_difficulty_paraphrases:
                    num_simple_phrases += 1
                else:
                    num_draw_phrases += 1
        return num_difficult_phrases, num_simple_phrases, num_draw_phrases

    def _make_n_gram_phrase(self, sequence, n=2):
        return (' '.join(sequence[i:i + n]) for i in range(len(sequence) - (n - 1)))


class WordBasedDifficulty:
    def __init__(self, device, is_negative=False):
        self.device = device
        self.lexicon_path = "/research/king3/hpchan/projects/seq-gen-mdp/word_complexity_lexicon/word_to_complexity.pkl"
        with open(self.lexicon_path, 'rb') as f:
            self.word_to_complexity = pkl.load(f)
        self.stopwords = stopwords.words('english')
        self.is_negative = is_negative

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        batch_cost = []
        for pred_word_list in pred_word_2d_list:
            complexity_scores = []
            for pred_word in pred_word_list:
                complexity_score = self.word_to_complexity[pred_word]
                if complexity_score > 0 and pred_word not in self.stopwords:
                    complexity_scores.append(complexity_score)
                    print("{}: {}".format(pred_word, complexity_score))
            if len(complexity_scores) > 0:
                avg_complexity_score = sum(complexity_scores) / len(complexity_scores)
            else:
                avg_complexity_score = 3.5
            if self.is_negative:
                avg_complexity_score = 0.0 - avg_complexity_score
            batch_cost.append(avg_complexity_score)

        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class WikiRelativeWordFrequency:
    def __init__(self, device, is_simple=False):
        self.device = device
        self.all_wiki_vocab_path = "/research/king3/hpchan/datasets/wikipedia-aligned.v2/all_wiki_vocab_cnt_normalized.pkl"
        self.normal_wiki_vocab_path = "/research/king3/hpchan/datasets/wikipedia-aligned.v2/normal_wiki_vocab_cnt_normalized.pkl"
        self.simple_wiki_vocab_path = "/research/king3/hpchan/datasets/wikipedia-aligned.v2/simple_wiki_vocab_cnt_normalized.pkl"
        with open(self.all_wiki_vocab_path, 'rb') as f:
            self.all_wiki_vocab = pkl.load(f)
        self.is_simple = is_simple

        if is_simple:
            target_wiki_vocab_path = self.simple_wiki_vocab_path
        else:
            target_wiki_vocab_path = self.normal_wiki_vocab_path

        with open(target_wiki_vocab_path, 'rb') as f:
            self.target_wiki_vocab = pkl.load(f)

        self.stopwords = stopwords.words('english')

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
              trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        batch_cost = []
        for pred_word_list in pred_word_2d_list:
            relative_frequencies = []
            for pred_word in pred_word_list:
                frequency_all = self.all_wiki_vocab[pred_word]
                if frequency_all > 0 and pred_word not in self.stopwords:
                    frequency_ref = self.target_wiki_vocab[pred_word]
                    relative_frequency = frequency_ref / frequency_all
                    relative_frequencies.append(relative_frequency)
                    print('{}: {}'.format(pred_word, relative_frequency))

            if len(relative_frequencies) > 0:
                avg_scores = sum(relative_frequencies) / len(relative_frequencies)
            else:
                avg_scores = 1.0

            batch_cost.append(avg_scores)

        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


class CorruptedDiscriminatorCost:
    def __init__(self, device):
        self.device = device
        model_dir = "saved_model/bert_corrupted_xsum_classifier.bert_classifier.20191101-112951"
        ckpt_dir = model_dir + "/ckpt/epoch-4-total_batch-44000-valid_f1-0.9975"
        self.discriminator_model = SeqClassifyDiscriminator(model_dir, ckpt_dir, device=device)

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        pred_str_list = [' '.join(pred_word_list) for pred_word_list in pred_word_2d_list]
        class_prob = self.discriminator_model.score(pred_str_list)  # [batch, 2]
        return class_prob[:, 1]  # [batch]

def compute_batch_cost(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, cost_objs, control_variables):
    batch_cost_list = []
    for cost_obj in cost_objs:
        cost = cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        batch_cost_list.append(cost)  # a list of tensor with size [batch_size]
    batch_cost_2d = torch.stack(batch_cost_list, dim=1)  # tensor: [batch, num_cost_types]
    return batch_cost_2d


class BadEndingCost:
    def __init__(self, device):
        self.device = device
        self.end_tokens = ['.', '!', '?', '...', "'", "`", '"', ")"]
        self.article_tokens = ['a', 'an', 'the']
        #self.stop_words = set(stopwords.words('english'))

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        batch_cost = []
        for pred_word_list, pred_sent_list in zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized):
            if len(pred_word_list) > 0:
                if self.has_ending_punctuation(pred_word_list):
                    if self.end_with_article(pred_sent_list):
                        cost = 1.0
                    else:
                        cost = 0.0
                else:
                    cost = 1.0
            else:
                cost = 0.0
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)

    def has_ending_punctuation(self, pred_word_list):
        return pred_word_list[-1] in self.end_tokens

    def end_with_article(self, pred_sent_list):
        for pred_sent_word_list in pred_sent_list:
            if len(pred_sent_word_list) > 1 and pred_sent_word_list[-2] in self.article_tokens:
                return True
        return False

"""
def compute_batch_cost(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, cost_types, device):
    with torch.no_grad():
        #num_cost_types = len(cost_types)
        batch_cost_list = []
        #batch_cost_2d = np.zeros((batch_size, num_cost_types))
        for i, cost_type in enumerate(cost_types):
            if cost_type == 0:
                cost_func = has_three_gram_repeat
            elif cost_type == 1:
                cost_func = min_len_cost
            else:
                raise ValueError("No matched cost function type.")
            batch_cost_list.append(cost_func(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device))
            #batch_cost_2d[:, i] = cost_func(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size)

        batch_cost_2d = torch.stack(batch_cost_list, dim=1)  # tensor: [batch, num_cost_types]
    return batch_cost_2d
"""

def num_three_gram_repeat(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    # return the number of 3-gram repeat for each prediciton sequence in the batch
    batch_cost = np.zeros(batch_size)
    for batch_i, pred_str in enumerate(pred_str_list):
        three_gram_counter = Counter()
        three_gram_counter.update(_make_n_gram(pred_str, n=3))
        three_gram_repeat = sum(c - 1 for g, c in three_gram_counter.items() if c > 1)
        batch_cost[batch_i] = three_gram_repeat
    return torch.from_numpy(batch_cost).type(torch.FloatTensor).to(device)  # tensor: [batch_size]


class ThreeGramRepeatLoss:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        batch_cost = []
        for batch_i, pred_word_list in enumerate(pred_word_2d_list):
            three_gram_counter = Counter()
            three_gram_counter.update(_make_n_gram(pred_word_list, n=3))
            three_gram_repeat = sum(c - 1 for g, c in three_gram_counter.items() if c > 1)
            if three_gram_repeat > 0:
                cost = 1.0
            else:
                cost = 0.0
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)


class ThreeGramRepeatFraction:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        batch_cost = []
        for batch_i, pred_word_list in enumerate(pred_word_2d_list):
            three_gram_counter = Counter()
            three_gram_counter.update(_make_n_gram(pred_word_list, n=3))
            num_three_gram_repeat = sum(c - 1 for g, c in three_gram_counter.items() if c > 1)
            num_pred_three_grams = sum(three_gram_counter.values())
            if num_pred_three_grams > 0:
                three_gram_repeat_fraction = num_three_gram_repeat / num_pred_three_grams
            else:
                three_gram_repeat_fraction = 1
            batch_cost.append(three_gram_repeat_fraction)
        return torch.FloatTensor(batch_cost).to(self.device)


class ThreeGramNoveltyBinDistance:
    def __init__(self, device, n=3):
        self.device = device
        self.n = n

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']
        batch_cost = []
        for pred_word_list, src_word_list, target_abs_bin in zip(pred_word_2d_list, src_word_2d_list, target_abs_bins):
            if len(pred_word_list) >= self.n:
                n_gram_novelty = compute_n_gram_novelty(pred_word_list, src_word_list, n=self.n)
                abs_bin = n_gram_novelty_to_bin(n_gram_novelty)
                abs_bin_distance = abs(abs_bin - target_abs_bin) / 3
            else:
                abs_bin_distance = 1.0
            batch_cost.append(abs_bin_distance)
        return torch.FloatTensor(batch_cost).to(self.device)


class ExtFragDensityBinDistance:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']
        batch_cost = []
        for pred_word_list, src_word_list, target_abs_bin in zip(pred_word_2d_list, src_word_2d_list, target_abs_bins):
            if len(pred_word_list) >= 1:
                ext_fragment_list = compute_extractive_fragment(src_word_list, pred_word_list)
                ext_frag_density = compute_extractive_fragment_density(ext_fragment_list, pred_word_list)
                abs_bin = ext_frag_density_to_bin(ext_frag_density)
                abs_bin_distance = abs(abs_bin - target_abs_bin) / 3
            else:
                abs_bin_distance = 1.0
            batch_cost.append(abs_bin_distance)
        return torch.FloatTensor(batch_cost).to(self.device)


class ComputeExtFragDensityBin:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']
        batch_cost = []
        for pred_word_list, src_word_list, target_abs_bin in zip(pred_word_2d_list, src_word_2d_list, target_abs_bins):
            if len(pred_word_list) >= 1:
                ext_fragment_list = compute_extractive_fragment(src_word_list, pred_word_list)
                ext_frag_density = compute_extractive_fragment_density(ext_fragment_list, pred_word_list)
                abs_bin = ext_frag_density_to_bin(ext_frag_density)
                abs_bin_distance = abs(abs_bin - target_abs_bin) / 3
            else:
                abs_bin_distance = 1.0
            batch_cost.append(abs_bin_distance)
        return torch.FloatTensor(batch_cost).to(self.device)


class ReferencePolicy:
    def __init__(self, device, model, TLDR_id_list, pad_idx, eos_idx):
        self.device = device
        self.model = model
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.TLDR_id_list = TLDR_id_list
        self.model.eval()
        #print("pad_idx: {}".format(pad_idx))
        #print("eos_idx: {}".format(eos_idx))
        #print("TLDR_id-list: {}".format(TLDR_id_list))

    def analyze_prediction(self, prompt_ids_list, pred_ids_list):
        """
        :param prompt_ids_list: truncated_doc + tldr_ids_list
        :param pred_ids_list: not include eos
        :return:
        """
        input_ids_list = []
        label_ids_list = []
        for prompt_ids, pred_ids in zip(prompt_ids_list, pred_ids_list):
            input_ids = prompt_ids + pred_ids + [self.eos_idx]
            input_ids_list.append(input_ids)
            label_ids = [-100] * len(prompt_ids) + pred_ids + [self.eos_idx]
            label_ids_list.append(label_ids)
        max_input_ids_len = max([len(input_ids) for input_ids in input_ids_list])
        input_ids_list_padded = []
        label_ids_list_padded = []
        for input_ids, label_ids in zip(input_ids_list, label_ids_list):
            padding_length = max_input_ids_len - len(input_ids)
            input_ids_list_padded.append(input_ids + [self.pad_idx] * padding_length)
            label_ids_list_padded.append(label_ids + [-100] * padding_length)
        #print("input_ids sizes: ")
        #[print(len(input_ids))for input_ids in input_ids_list_padded]
        #[print(len(label_ids)) for label_ids in label_ids_list_padded]
        #print("pred_ids sizes: ")
        #[print(len(pred_ids)) for pred_ids in pred_ids_list]
        input_ids_tensor = torch.LongTensor(input_ids_list_padded).to(self.device)
        label_ids_tensor = torch.LongTensor(label_ids_list_padded).to(self.device)
        mask = torch.ne(input_ids_tensor, self.pad_idx).float()
        outputs = self.model(input_ids=input_ids_tensor, attention_mask=mask, labels=label_ids_tensor)
        ref_log_probs = outputs[0]  # [batch_size]
        return ref_log_probs


class KLDivergenceCost:
    def __init__(self, device, reference_policy):
        self.device = device
        self.reference_policy = reference_policy

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        pred_ids_list = control_variables['pred_ids_list']  # not include eos
        prompt_ids_list = control_variables['prompt_ids_list']  # doc_truncated + tldr_ids
        pred_log_probs_sum = control_variables['pred_log_probs_sum']  # [batch]
        pred_seq_lens_tensor = torch.FloatTensor([len(pred_ids) + 1 for pred_ids in pred_ids_list]).to(self.device) # +1 to include eos
        with torch.no_grad():
            ref_log_probs_sum = self.reference_policy.analyze_prediction(prompt_ids_list, pred_ids_list)  # [batch]
            kl_divergence = pred_log_probs_sum - ref_log_probs_sum
            normalized_kl_divergence = kl_divergence / pred_seq_lens_tensor
        #print("kl_divergence: ")
        #print(normalized_kl_divergence)
        return normalized_kl_divergence


class ExtFragDensityBinDistanceParallel:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']

        arg_list = list(zip(pred_word_2d_list, src_word_2d_list, target_abs_bins))

        with mp.Pool() as pool:
            batch_cost_list = list(pool.map(self.score_, arg_list))

        return torch.FloatTensor(batch_cost_list).to(self.device)

    def _score(self, args):
        pred_word_list, src_word_list, target_abs_bin = args
        if len(pred_word_list) >= 1:
            ext_fragment_list = compute_extractive_fragment(src_word_list, pred_word_list)
            ext_frag_density = compute_extractive_fragment_density(ext_fragment_list, pred_word_list)
            abs_bin = ext_frag_density_to_bin(ext_frag_density)
            abs_bin_distance = abs(abs_bin - target_abs_bin) / 3
        else:
            abs_bin_distance = 1.0
        return abs_bin_distance

class SentenceFusionBinDistance:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        #start_time = time.time()
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        #src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']
        src_word_2d_list_sent_tokenized = control_variables['src_word_2d_list_sent_tokenized']
        batch_cost = []
        for pred_word_list, pred_word_list_sent_tokenized, src_word_list_sent_tokenized, target_abs_bin in zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, src_word_2d_list_sent_tokenized, target_abs_bins):
            num_summary_sents = len(pred_word_list_sent_tokenized)
            if len(pred_word_list) >= 1 and num_summary_sents > 0:
                #print(pred_word_list)
                #print(pred_word_list_sent_tokenized)
                #print(src_word_list_sent_tokenized[:3])
                simple_similar_source_indices, lcs_paths_list, smooth_article_paths_list = ssi_functions.get_simple_source_indices_list(
                    pred_word_list_sent_tokenized, src_word_list_sent_tokenized, vocab=None, sentence_limit=3,
                    min_matched_tokens=2)
                num_source_sentences = 0
                for source_indices in simple_similar_source_indices:
                    source_indices_len = len(source_indices)
                    if source_indices_len == 0:
                        source_indices_len = 3
                    num_source_sentences += source_indices_len
                avg_fusion_ratio = num_source_sentences / num_summary_sents
                #print(num_summary_sents)
                #print(num_source_sentences)
                #print(avg_fusion_ratio)
                abs_bin = fusion_ratio_to_bin(avg_fusion_ratio)
                abs_bin_distance = abs(abs_bin - target_abs_bin) / 3
                #print(abs_bin)
            else:
                abs_bin_distance = 1.0
            batch_cost.append(abs_bin_distance)
        #score_time = time_since(start_time)
        #print("score time: {}".format(score_time))
        return torch.FloatTensor(batch_cost).to(self.device)


class SentenceFusionBinDistanceParallel:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        #src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']
        src_word_2d_list_sent_tokenized = control_variables['src_word_2d_list_sent_tokenized']

        # construct a list of arguments
        arg_list = list(zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, src_word_2d_list_sent_tokenized, target_abs_bins))

        with mp.Pool() as pool:
            batch_cost_list = list(pool.map(self.score_, arg_list))

        return torch.FloatTensor(batch_cost_list).to(self.device)

    def score_(self, args):
        pred_word_list, pred_word_list_sent_tokenized, src_word_list_sent_tokenized, target_abs_bin = args
        if len(pred_word_list) >= 1:
            # print(pred_word_list)
            # print(pred_word_list_sent_tokenized)
            # print(src_word_list_sent_tokenized[:3])
            simple_similar_source_indices, lcs_paths_list, smooth_article_paths_list = ssi_functions.get_simple_source_indices_list(
                pred_word_list_sent_tokenized, src_word_list_sent_tokenized, vocab=None, sentence_limit=3,
                min_matched_tokens=2)
            num_source_sentences = 0
            num_summary_sents = len(pred_word_list_sent_tokenized)
            for source_indices in simple_similar_source_indices:
                source_indices_len = len(source_indices)
                if source_indices_len == 0:
                    source_indices_len = 3
                num_source_sentences += source_indices_len
            avg_fusion_ratio = num_source_sentences / num_summary_sents
            # print(num_summary_sents)
            # print(num_source_sentences)
            # print(avg_fusion_ratio)
            abs_bin = fusion_ratio_to_bin(avg_fusion_ratio)
            abs_bin_distance = abs(abs_bin - target_abs_bin) / 3
            # print(abs_bin)
        else:
            abs_bin_distance = 1.0
        return abs_bin_distance


class SentenceFusionBinDistanceMultiProcess:
    def __init__(self, device, max_batch_size):
        self.device = device
        self.shared_result_array = mp.Array('d', max_batch_size)
        self.max_batch_size = max_batch_size

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        #start_time = time.time()
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        #src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']
        src_word_2d_list_sent_tokenized = control_variables['src_word_2d_list_sent_tokenized']

        # construct a list of arguments and a shared memory
        arg_list = list(zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, src_word_2d_list_sent_tokenized, target_abs_bins))
        #shared_result_array = mp.Array('d', batch_size)
        processes = []
        num_cpus = 2

        # compute the steps
        if batch_size < num_cpus:
            num_samples_per_cpu = 1
        else:
            if batch_size % num_cpus == 0:
                num_samples_per_cpu = batch_size // num_cpus
            else:
                num_samples_per_cpu = batch_size // num_cpus + 1
        num_processed_samples = 0

        #print("num_samples_per_cpu: {}".format(num_samples_per_cpu))
        #print("batch_size: {}".format(batch_size))
        #print("num_cpus: {}".format(num_cpus))

        # instantiating process with arguments
        for i in range(num_cpus):
            num_samples_to_process = min(num_samples_per_cpu, batch_size - num_processed_samples)
            idx_to_process = list(range(num_processed_samples, num_processed_samples+num_samples_to_process))
            #print("cpu: {}".format(i))
            #print("num_samples_to_process: {}".format(num_samples_to_process))
            #print("idx_to_process: {}".format(idx_to_process))
            p = mp.Process(target=self.score_, args=(arg_list[num_processed_samples:num_processed_samples+num_samples_to_process], self.shared_result_array, idx_to_process))
            processes.append(p)
            p.start()
            num_processed_samples += num_samples_to_process
            if num_processed_samples == batch_size:
                break

        # complete the processes
        for p in processes:
            p.join()

        result_list = list(self.shared_result_array[:batch_size])

        #print(result_list)
        #exit()
        #score_time = time_since(start_time)
        #print("score time: {}".format(score_time))
        return torch.FloatTensor(result_list).to(self.device)

    def score_(self, args, shared_result_array, shared_result_array_ids):
        num_processed_samples = 0
        for pred_word_list, pred_word_list_sent_tokenized, src_word_list_sent_tokenized, target_abs_bin in args:
            if len(pred_word_list) >= 1:
                # print(pred_word_list)
                # print(pred_word_list_sent_tokenized)
                # print(src_word_list_sent_tokenized[:3])
                simple_similar_source_indices, lcs_paths_list, smooth_article_paths_list = ssi_functions.get_simple_source_indices_list(
                    pred_word_list_sent_tokenized, src_word_list_sent_tokenized, vocab=None, sentence_limit=3,
                    min_matched_tokens=2)
                num_source_sentences = 0
                num_summary_sents = len(pred_word_list_sent_tokenized)
                for source_indices in simple_similar_source_indices:
                    source_indices_len = len(source_indices)
                    if source_indices_len == 0:
                        source_indices_len = 3
                    num_source_sentences += source_indices_len
                avg_fusion_ratio = num_source_sentences / num_summary_sents
                # print(num_summary_sents)
                # print(num_source_sentences)
                # print(avg_fusion_ratio)
                abs_bin = fusion_ratio_to_bin(avg_fusion_ratio)
                abs_bin_distance = abs(abs_bin - target_abs_bin) / 3
                # print(abs_bin)
            else:
                abs_bin_distance = 1.0
            current_processing_id = shared_result_array_ids[num_processed_samples]
            shared_result_array[current_processing_id] = abs_bin_distance
            num_processed_samples += 1

        return abs_bin_distance


class NovelNgramFraction:
    def __init__(self, device, n=3, is_negative=False):
        self.device = device
        self.n = n
        self.is_negative = is_negative

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        src_word_2d_list = control_variables['src_word_2d_list']
        batch_cost = []
        for pred_word_list, src_word_list in zip(pred_word_2d_list, src_word_2d_list):
            pred_n_gram_counter = Counter()
            pred_n_gram_counter.update(_make_n_gram(pred_word_list, n=self.n))

            src_n_gram_counter = Counter()
            src_n_gram_counter.update(_make_n_gram(src_word_list, n=self.n))

            all_pred_n_grams = pred_n_gram_counter.keys()
            num_unique_pred_n_grams = len(all_pred_n_grams)
            if num_unique_pred_n_grams > 0:
                num_unique_novel_pred_n_grams = 0
                for n_gram in all_pred_n_grams:
                    if n_gram not in src_n_gram_counter:
                        num_unique_novel_pred_n_grams += 1
                novel_n_gram_fraction = num_unique_novel_pred_n_grams / num_unique_pred_n_grams
            else:
                if self.is_negative:
                    novel_n_gram_fraction = 0.0
                else:
                    novel_n_gram_fraction = 1.0

            if self.is_negative:
                novel_n_gram_fraction = 0.0 - novel_n_gram_fraction

            batch_cost.append(novel_n_gram_fraction)
        return torch.FloatTensor(batch_cost).to(self.device)


class SentLevelNovelNgramFraction:
    def __init__(self, device, n=3, is_negative=False):
        self.device = device
        self.n = n
        self.is_negative = is_negative

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        # if the predicted sequence has repeat 3-gram, cost = 1, else 0
        src_word_2d_list = control_variables['src_word_2d_list']
        batch_cost = []
        for pred_word_list, src_word_list in zip(pred_word_2d_list_sent_tokenized, src_word_2d_list):
            pred_n_gram_counter = Counter()
            pred_n_gram_counter.update(_make_n_gram(pred_word_list, n=self.n))

            src_n_gram_counter = Counter()
            src_n_gram_counter.update(_make_n_gram(src_word_list, n=self.n))

            all_pred_n_grams = pred_n_gram_counter.keys()
            num_unique_pred_n_grams = len(all_pred_n_grams)
            if num_unique_pred_n_grams > 0:
                num_unique_novel_pred_n_grams = 0
                for n_gram in all_pred_n_grams:
                    if n_gram not in src_n_gram_counter:
                        num_unique_novel_pred_n_grams += 1
                novel_n_gram_fraction = num_unique_novel_pred_n_grams / num_unique_pred_n_grams
            else:
                if self.is_negative:
                    novel_n_gram_fraction = 0.0
                else:
                    novel_n_gram_fraction = 1.0

            if self.is_negative:
                novel_n_gram_fraction = 0.0 - novel_n_gram_fraction

            batch_cost.append(novel_n_gram_fraction)
        return torch.FloatTensor(batch_cost).to(self.device)


class IncorrectBut:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        batch_cost = []
        for pred_word_list, trg_word_list in zip(pred_word_2d_list, trg_word_2d_list):
            # search for but
            pred_but = search_for_word(pred_word_list, "but")
            trg_but = search_for_word(trg_word_list, "but")
            if pred_but and not trg_but:
                cost = 1.0
            else:
                cost = 0.0
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


def search_for_word(word_list, trg_word):
    trg_word = trg_word.lower()
    for w in word_list:
        if w.lower() == trg_word:
            return True
    return False


def has_three_gram_repeat(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    # return the number of 3-gram repeat for each prediciton sequence in the batch
    #batch_cost = np.zeros(batch_size)
    batch_cost = []
    for batch_i, pred_str in enumerate(pred_str_list):
        three_gram_counter = Counter()
        three_gram_counter.update(_make_n_gram(pred_str, n=3))
        three_gram_repeat = sum(c - 1 for g, c in three_gram_counter.items() if c > 1)
        if three_gram_repeat > 0:
            cost = 1.0
        else:
            cost = 0.0
        batch_cost.append(cost)
    return torch.FloatTensor(batch_cost).to(device)  # tensor: [batch_size]


class MinLenCost:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        batch_cost = []
        for batch_i, pred_word_list in enumerate(pred_word_2d_list):
            if len(pred_word_list) < 10:
                cost = 1.0
            else:
                cost = 0.0
            batch_cost.append(cost)
        return torch.FloatTensor(batch_cost).to(self.device)  # tensor: [batch_size]


def min_len_cost(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, batch_size, device):
    batch_cost = []
    for batch_i, pred_str in enumerate(pred_str_list):
        if len(pred_str) < 10:
            cost = 1.0
        else:
            cost = 0.0
        batch_cost.append(cost)
    return torch.FloatTensor(batch_cost).to(device)  # tensor: [batch_size]


if __name__ == "__main__":
    #cost_obj = ThreeGramRepeatFraction(device='cpu')
    #pred_word_2d_list = [['this', 'is', 'good', '.'], []]
    #scores = cost_obj.score(pred_word_2d_list, None, None, None, None, None)
    #print(scores)

    #cost_obj = NegativeNamedEntityF1(device="cpu")
    #pred_word_2d_list = [['donald', 'trump', 'beats', 'david', 'in', 'boston', '.'], []]
    #control_variables = {'reference_entities': [['donald trump', 'david beckham', 'singapore', 'boston'], ['china', 'japan']]}
    #scores = cost_obj.score(pred_word_2d_list, None, None, None, None, control_variables)
    #print(scores)
    split_dir = "/mnt/sharedfolder/hpchan/datasets/cased-cnn-dailymail_coref_3/train"
    cost_obj = NegativeNamedEntityQAF1(device="cuda:0", threshold=0.9)
    masked_questions_ids_2dlist = []
    answer_2dlist = []
    summary_word_2d_list = []
    for i in range(3):
        with open(os.path.join(split_dir, "{}.json".format(i))) as f:
            js = json.loads(f.read())
        masked_questions_ids_list = js["masked_question_ids_list"]
        answer_list = js["answer_list"]
        summary_sent_list = js["abstract"]
        summary_str = ' '.join(summary_sent_list)
        summary_word_list = summary_str.split(" ")
        masked_questions_ids_2dlist.append(masked_questions_ids_list)
        answer_2dlist.append(answer_list)
        summary_word_2d_list.append(summary_word_list)

    control_variables = {}
    control_variables['masked_questions_ids_2dlist'] = masked_questions_ids_2dlist
    control_variables['answer_2dlist'] = answer_2dlist
    scores = cost_obj.score(summary_word_2d_list, None, None, None, None, control_variables)
