import numpy as np
from utils.string_helper import *
import torch
from metric import compute_rouge_l_summ, compute_rouge_n, compute_rouge_l, compute_rouge_n_zipped, compute_weighted_rouge_l_summ
from dataset_extractive_fragment_stat import compute_extractive_fragment, compute_extractive_fragment_density
import multiprocessing as mp
from bert_score import BERTScorer
from utils.io import ext_frag_density_to_bin
from collections import defaultdict
from transformers import AutoTokenizer

from bert_score.utils import (get_idf_dict, bert_cos_score_idf,
                    get_bert_embedding, model_types,
                    lang2model, model2layers, get_hash)

from bert_score.score import get_model
from utils.cost import LengthBinConsistent, LengthBinDistance, ThreeGramRepeatFraction, ExtFragDensityBinDistance, IncorrectBut, NegativeNamedEntityQAF1, EntityRepeatCost
from cytoolz import concat
import sys


def sample_list_to_str_list(sample_list, oov_lists, idx2word, vocab_size, eos_idx, unk_idx=None, replace_unk=False, src_str_list=None):
    """Convert a list of sample dict to a 2d list of predicted keyphrases"""
    pred_word_2d_list = []  #  a 2dlist, len(pred_str_2d_list)=batch_size, len(pred_str_2d_list[0])=
    for sample, oov, src_word_list in zip(sample_list, oov_lists, src_str_list):
        # sample['prediction']: list of 0-dim tensor, len=trg_len
        # sample['attention']: tensor with size [trg_len, src_len]
        pred_word_list = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, sample['attention'])
        pred_word_2d_list.append(pred_word_list)
    return pred_word_2d_list


def compute_batch_reward(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, reward_obj, regularization_factor=0.0, regularization_type=0, entropy=None, control_variables=None):
    batch_reward = reward_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables)
    return batch_reward  # tensor: [batch_size]


def compute_batch_reward_backup(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, reward_type, regularization_factor=0.0, regularization_type=0, entropy=None, device='cpu'):
    with torch.no_grad():
        # A reward function returns a tensor of size [batch_size]
        if reward_type == 0:
            reward_func = xsum_mixed_rouge_reward
        elif reward_type == 1:
            reward_func = rouge_l_reward
        return reward_func(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, device)


class SummRougeLReward:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        reward = []
        for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
                zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized)):
            reward.append(compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f'))
        return torch.FloatTensor(reward).to(self.device)  # tensor: [batch_size]


class RougeLReward:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        reward = []
        for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
                zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized)):
            reward.append(compute_rouge_l(pred_str, trg_str, mode='f'))
        return torch.FloatTensor(reward).to(self.device)  # tensor: [batch_size]


class GOLCReward:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        reward = []
        for idx, (pred_word_list, pred_sent_list, trg_word_list, trg_sent_list) in enumerate(
                zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized)):
            # trim prediction
            target_word_len = len(trg_word_list)
            pred_word_len = len(pred_word_list)
            truncated_pred_word_list = pred_word_list[:target_word_len]
            rouge_l_recall = compute_rouge_l(truncated_pred_word_list, trg_word_list, mode='r')
            reward.append(rouge_l_recall - max(0, pred_word_len - target_word_len) )
        return torch.FloatTensor(reward).to(self.device)  # tensor: [batch_size]


class Rouge2Reward:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
              trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        reward = []
        for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
                zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
                    trg_word_2d_list_sent_tokenized)):
            reward.append(compute_rouge_n(pred_str, trg_str, n=2, mode='f'))
        return torch.FloatTensor(reward).to(self.device)  # tensor: [batch_size]


class MixedRougeRewardLenPenalty:
    def __init__(self, device):
        self.device = device
        self.len_bin_cost_obj = LengthBinConsistent(device)

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
              trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        reward = []
        for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
                zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
                    trg_word_2d_list_sent_tokenized)):
            reward.append(
                0.2 * compute_rouge_n(pred_str, trg_str, n=1, mode='f') + 0.5 * compute_rouge_n(pred_str, trg_str, n=2,
                                                                                                mode='f') + 0.3 * compute_rouge_l_summ(
                    pred_sent_list, trg_sent_list, mode='f'))
        reward_tensor = torch.FloatTensor(reward).to(self.device)
        penalty_tensor = self.len_bin_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list,
              trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        return reward_tensor - penalty_tensor


class MixedRougeReward:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        rewards = []
        for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
                zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized)):
            #reward = 0.2 * compute_rouge_n(pred_str, trg_str, n=1, mode='f') + 0.5 * compute_rouge_n(pred_str, trg_str, n=2, mode='f') + 0.3 * compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f')
            reward = 0.3333333 * compute_rouge_n(pred_str, trg_str, n=1, mode='f') + 0.3333333 * compute_rouge_n(pred_str, trg_str, n=2, mode='f') + 0.3333333 * compute_rouge_l(pred_str, trg_str, mode='f')
            rewards.append(reward)
        return torch.FloatTensor(rewards).to(self.device)  # tensor: [batch_size]


class WeightedROUGELReward:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        rewards = []
        reference_entities = control_variables["reference_entities_list"]
        for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list, reference_entity_list) in enumerate(
                zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, reference_entities)):
            #print("reference_entity_list")
            #print(reference_entity_list)
            important_word_list = " ".join(reference_entity_list).split(" ")
            #print("important_word_list")
            #print(important_word_list)
            #print("pred_sent_list")
            #print(pred_sent_list)
            #print("trg_sent_list")
            #print(trg_sent_list)
            reward = compute_weighted_rouge_l_summ(pred_sent_list, trg_sent_list, important_word_list, mode='f')
            #print("reward")
            #print(reward)
            rewards.append(reward)
        return torch.FloatTensor(rewards).to(self.device)  # tensor: [batch_size]


class Rouge2LReward:
    def __init__(self, device):
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        rewards = []
        for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
                zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized)):
            reward = 0.5 * compute_rouge_n(pred_str, trg_str, n=2, mode='f') + 0.5 * compute_rouge_l(pred_str, trg_str, mode='f')
            rewards.append(reward)
        return torch.FloatTensor(rewards).to(self.device)  # tensor: [batch_size]


class BertRougeReward:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.num_layers = model2layers[self.model_type]
        self.model = get_model(self.model_type, num_layers=self.num_layers, all_layers=False)
        self.device = device
        self.model.to(device)

        # do not use idf
        self.idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0
        self.max_token = 200

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        batch_size = len(cands)
        try:
            all_preds = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, self.idf_dict,
                                   verbose=False, device=self.device,
                                   batch_size=batch_size, all_layers=False)
        except:
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        F1 = all_preds[..., 2].to(self.device)  # [batch_size]

        # compute ROUGE-L
        RougeLs = []
        for pred_word_list, trg_word_list in zip(pred_word_2d_list, trg_word_2d_list):
            RougeLs.append(compute_rouge_l(pred_word_list, trg_word_list, mode='f'))
        RougeLs = torch.FloatTensor(RougeLs).to(self.device)  # [batch_size]
        return F1 * 0.5 + RougeLs * 0.5


class BertRouge2Reward:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.num_layers = model2layers[self.model_type]
        self.model = get_model(self.model_type, num_layers=self.num_layers, all_layers=False)
        self.device = device
        self.model.to(device)

        # do not use idf
        self.idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0
        self.max_token = 200

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        batch_size = len(cands)
        try:
            all_preds = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, self.idf_dict,
                                   verbose=False, device=self.device,
                                   batch_size=batch_size, all_layers=False)
        except:
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        F1 = all_preds[..., 2].to(self.device)  # [batch_size]

        # compute ROUGE-L
        Rouge2s = []
        for pred_word_list, trg_word_list in zip(pred_word_2d_list, trg_word_2d_list):
            Rouge2s.append(compute_rouge_n(pred_word_list, trg_word_list, n=2, mode='f'))
        Rouge2s = torch.FloatTensor(Rouge2s).to(self.device)  # [batch_size]
        return F1 * 0.5 + Rouge2s * 0.5


class BertRouge2RewardParallel:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.num_layers = model2layers[self.model_type]
        self.model = get_model(self.model_type, num_layers=self.num_layers, all_layers=False)
        self.device = device
        self.model.to(device)

        # do not use idf
        self.idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0
        self.max_token = 200

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        batch_size = len(cands)
        try:
            all_preds = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, self.idf_dict,
                                   verbose=False, device=self.device,
                                   batch_size=batch_size, all_layers=False)
        except:
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        F1 = all_preds[..., 2].to(self.device)  # [batch_size]

        # compute ROUGE-L
        rouge_arg_list = list(zip(pred_word_2d_list, trg_word_2d_list))

        with mp.Pool() as pool:
            batch_rouge2_list = list(pool.map(compute_rouge_n_zipped(n=2, mode='f'), rouge_arg_list))

        Rouge2s = torch.FloatTensor(batch_rouge2_list).to(self.device)  # [batch_size]
        return F1 * 0.5 + Rouge2s * 0.5


def xsum_mixed_rouge_reward(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, device):
    #reward = np.zeros(batch_size)
    reward = []
    for idx, (pred_str, pred_sent_list, trg_str, trg_sent_list) in enumerate(
            zip(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized)):
        #reward[idx] = 0.2 * compute_rouge_n(pred_str, trg_str, n=1, mode='f') + 0.5 * compute_rouge_n(pred_str, trg_str, n=2, mode='f') + 0.3 * compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f')
        reward.append(0.2 * compute_rouge_n(pred_str, trg_str, n=1, mode='f') + 0.5 * compute_rouge_n(pred_str, trg_str, n=2, mode='f') + 0.3 * compute_rouge_l_summ(pred_sent_list, trg_sent_list, mode='f'))
    return torch.FloatTensor(reward).to(device)  # tensor: [batch_size]


class BertScoreReward:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.num_layers = model2layers[self.model_type]
        self.model = get_model(self.model_type, num_layers=self.num_layers, all_layers=False)
        self.device = device
        self.model.to(device)

        # do not use idf
        self.idf_dict = defaultdict(lambda: 1.)
        # set idf for [SEP] and [CLS] to 0
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0
        self.max_token = 200

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        batch_size = len(cands)
        try:
            all_preds = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, self.idf_dict,
                                   verbose=False, device=self.device,
                                   batch_size=batch_size, all_layers=False)
        except:
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        F1 = all_preds[..., 2]
        return F1.to(self.device)


class BertScoreRewardRescaled:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200
        self.device = device

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        P, R, F1 = self.scorer.score(cands, refs)
        """
        try:
            P, R, F1 = self.scorer.score(cands, refs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        """
        return F1.to(self.device)


class BertScoreRewardRescaledLengthPenalty:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200
        self.device = device
        self.len_bin_cost_obj = LengthBinDistance(device)

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        _, _, F1 = self.scorer.score(cands, refs)
        """
        try:
            P, R, F1 = self.scorer.score(cands, refs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        """
        # compute bert score
        reward_tensor = F1.to(self.device)
        # compute length bin distance
        penalty_tensor = self.len_bin_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        return reward_tensor - penalty_tensor


class BertScoreRewardRescaledLengthRepeatPenalty:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200
        self.device = device
        self.len_bin_cost_obj = LengthBinDistance(device)
        self.three_gram_repeat_cost_obj = ThreeGramRepeatFraction(device)
        print("length and repeat penalty!!!")
        print()

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        _, _, F1 = self.scorer.score(cands, refs)
        """
        try:
            P, R, F1 = self.scorer.score(cands, refs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        """
        # compute bert score
        reward_tensor = F1.to(self.device)
        # compute length bin distance
        len_penalty_tensor = self.len_bin_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        repeat_penalty_tensor = self.three_gram_repeat_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        return reward_tensor - len_penalty_tensor - repeat_penalty_tensor


class BertScoreRewardRescaledLengthRepeatPenaltyWeighted:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200
        self.device = device
        self.len_bin_cost_obj = LengthBinDistance(device)
        self.three_gram_repeat_cost_obj = ThreeGramRepeatFraction(device)
        print("length and repeat penalty!!!")
        print()

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        _, _, F1 = self.scorer.score(cands, refs)
        """
        try:
            P, R, F1 = self.scorer.score(cands, refs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        """
        # compute bert score
        reward_tensor = F1.to(self.device)
        # compute length bin distance
        len_penalty_tensor = self.len_bin_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        repeat_penalty_tensor = self.three_gram_repeat_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        return reward_tensor - 0.6 * len_penalty_tensor - 0.8 * repeat_penalty_tensor


class BertScoreRewardRescaledLengthRepeatPenaltyWeightedNew:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200
        self.device = device
        self.len_bin_cost_obj = LengthBinDistance(device)
        self.three_gram_repeat_cost_obj = ThreeGramRepeatFraction(device)
        print("length and repeat penalty!!!")
        print()

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        _, _, F1 = self.scorer.score(cands, refs)
        """
        try:
            P, R, F1 = self.scorer.score(cands, refs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        """
        # compute bert score
        reward_tensor = F1.to(self.device)
        # compute length bin distance
        len_penalty_tensor = self.len_bin_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        repeat_penalty_tensor = self.three_gram_repeat_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        return reward_tensor - 0.4 * len_penalty_tensor - 0.6 * repeat_penalty_tensor


class BertScoreRewardRescaledLengthRepeatPenaltyWeightedNew2:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200
        self.device = device
        self.len_bin_cost_obj = LengthBinDistance(device)
        self.three_gram_repeat_cost_obj = ThreeGramRepeatFraction(device)
        print("length and repeat penalty!!!")
        print()

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        _, _, F1 = self.scorer.score(cands, refs)
        """
        try:
            P, R, F1 = self.scorer.score(cands, refs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        """
        # compute bert score
        reward_tensor = F1.to(self.device)
        # compute length bin distance
        len_penalty_tensor = self.len_bin_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        repeat_penalty_tensor = self.three_gram_repeat_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        return reward_tensor - 0.9 * len_penalty_tensor - 1.1 * repeat_penalty_tensor


class AbsWeightedROUGEBaseline:
    def __init__(self, device):
        self.device = device
        print("abs weighted baseline!!!")
        print()

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        total_reward_list = []
        src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']

        for pred_word_list, src_word_list, trg_word_list, target_abs_bin in zip(pred_word_2d_list, src_word_2d_list, trg_word_2d_list, target_abs_bins):
            rouge_l_reward = compute_rouge_l(pred_word_list, trg_word_list, mode='f')
            if len(pred_word_list) >= 1:
                ext_fragment_list = compute_extractive_fragment(src_word_list, pred_word_list)
                ext_frag_density = compute_extractive_fragment_density(ext_fragment_list, pred_word_list)
                abs_bin = ext_frag_density_to_bin(ext_frag_density)
                if abs_bin == target_abs_bin:
                    abs_reward = 1.0
                else:
                    abs_reward = 0.0
            else:
                abs_reward = 0.0
            total_reward = 0.9 * rouge_l_reward + 0.1 * abs_reward
            total_reward_list.append(total_reward)
        return torch.FloatTensor(total_reward_list).to(self.device)  # tensor: [batch_size]


class AbsWeightedBaseline:
    def __init__(self, device):
        self.device = device
        print("abs weighted baseline!!!")
        print()
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        abs_reward_list = []
        src_word_2d_list = control_variables['src_word_2d_list']
        target_abs_bins = control_variables['abs_bins']

        # compute BERTScore
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        _, _, F1 = self.scorer.score(cands, refs)
        bertscore_reward_tensor = F1.to(self.device)

        for pred_word_list, src_word_list, trg_word_list, target_abs_bin in zip(pred_word_2d_list, src_word_2d_list, trg_word_2d_list, target_abs_bins):
            if len(pred_word_list) >= 1:
                ext_fragment_list = compute_extractive_fragment(src_word_list, pred_word_list)
                ext_frag_density = compute_extractive_fragment_density(ext_fragment_list, pred_word_list)
                abs_bin = ext_frag_density_to_bin(ext_frag_density)
                if abs_bin == target_abs_bin:
                    abs_reward = 1.0
                else:
                    abs_reward = 0.0
            else:
                abs_reward = 0.0
            abs_reward_list.append(abs_reward)
        abs_reward_tensor = torch.FloatTensor(abs_reward_list).to(self.device)
        return 0.9 * bertscore_reward_tensor + 0.1 * abs_reward_tensor


class BertScoreRewardRescaledExtractiveRepeatButPenaltyWeighted:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200
        self.device = device
        self.abs_bin_cost_obj = ExtFragDensityBinDistance(device)
        self.three_gram_repeat_cost_obj = ThreeGramRepeatFraction(device)
        self.but_cost_obj = IncorrectBut(device)
        print("abs, repeat, and but penalty!!!")
        print()

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        _, _, F1 = self.scorer.score(cands, refs)
        """
        try:
            P, R, F1 = self.scorer.score(cands, refs)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("cand")
            [print(cand) for cand in cands]
            [print(len(cand.split(' '))) for cand in cands]
            print("ref")
            [print(ref) for ref in refs]
            [print(len(ref.split(' '))) for ref in refs]
            exit()
        """
        # compute bert score
        reward_tensor = F1.to(self.device)
        # compute length bin distance
        abs_penalty_tensor = self.abs_bin_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        repeat_penalty_tensor = self.three_gram_repeat_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        but_penalty_tensor = self.but_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        return reward_tensor - 0.2 * abs_penalty_tensor - 0.5 * repeat_penalty_tensor - 0.3 * but_penalty_tensor


class BertScoreRewardRescaledQAF1EntityRepeatPenaltyWeighted:
    def __init__(self, device):
        self.model_type = "bert-base-uncased"
        self.scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True, device=device)
        self.max_token = 200
        self.device = device
        self.negative_QAF1_cost_obj = NegativeNamedEntityQAF1(device, 1.0)
        self.three_gram_repeat_cost_obj = ThreeGramRepeatFraction(device)
        self.entity_repeat_cost_obj = EntityRepeatCost(device)
        print("QAF1, repeat, and entity repeat penalty!!!")
        print()

    def score(self, pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, control_variables):
        cands = [' '.join(pred_word_list[:self.max_token]) for pred_word_list in pred_word_2d_list]  # a list of str
        refs = [' '.join(trg_word_list[:self.max_token]) for trg_word_list in trg_word_2d_list]  # a list of str
        _, _, F1 = self.scorer.score(cands, refs)
        # compute bert score
        reward_tensor = F1.to(self.device)
        # compute length bin distance
        negative_QAF1_tensor = self.negative_QAF1_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        repeat_penalty_tensor = self.three_gram_repeat_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        entity_repeat_penalty_tensor = self.entity_repeat_cost_obj.score(pred_word_2d_list, pred_word_2d_list_sent_tokenized,
                                                     trg_word_2d_list,
                                                     trg_word_2d_list_sent_tokenized, batch_size, control_variables)
        return reward_tensor - 0.15 * negative_QAF1_tensor - 0.5 * repeat_penalty_tensor - 0.4 * entity_repeat_penalty_tensor


"""
def bert_score_reward_no_stop_word(pred_word_2d_list, pred_word_2d_list_sent_tokenized, trg_word_2d_list, trg_word_2d_list_sent_tokenized, batch_size, device):
    stop_words = stopwords.words('english')
    # remove stop words
    pred_word_2d_list = [w for w in pred_word_2d_list if w not in stop_words]
    trg_word_2d_list = [w for w in trg_word_2d_list if w not in stop_words]
    P, R, F1 = score(cands=' '.join(pred_word_2d_list), refs=' '.join(trg_word_2d_list), model_type="bert-base-uncased")
    return F1.to(device)
"""


def compute_phrase_reward(pred_str_2dlist, trg_str_2dlist, batch_size, max_num_phrases, reward_shaping, reward_type, topk, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    phrase_reward = np.zeros((batch_size, max_num_phrases))
    if reward_shaping:
        for t in range(max_num_phrases):
            pred_str_2dlist_at_t = [pred_word_2d_list[:t + 1] for pred_word_2d_list in pred_str_2dlist]
            phrase_reward[:, t] = compute_batch_reward(pred_str_2dlist_at_t, trg_str_2dlist, batch_size, reward_type, topk, match_type, regularization_factor, regularization_type, entropy)
    else:
        phrase_reward[:, -1] = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type,
                                                               topk, match_type, regularization_factor, regularization_type, entropy)
    return phrase_reward


def compute_phrase_reward_backup(pred_str_2dlist, trg_str_2dlist, batch_size, num_predictions, reward_shaping, reward_type, topk, match_type="exact", regularization_factor=0.0, regularization_type=0, entropy=None):
    phrase_reward = np.zeros((batch_size, num_predictions))
    if reward_shaping:
        for t in range(num_predictions):
            pred_str_2dlist_at_t = [pred_word_2d_list[:t + 1] for pred_word_2d_list in pred_str_2dlist]
            phrase_reward[:, t] = compute_batch_reward(pred_str_2dlist_at_t, trg_str_2dlist, batch_size, reward_type, topk, match_type, regularization_factor, regularization_type, entropy)
    else:
        phrase_reward[:, num_predictions - 1] = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type,
                                                               topk, match_type, regularization_factor, regularization_type, entropy)
    return phrase_reward


def shape_reward(reward_np_array):
    batch_size, seq_len = reward_np_array.shape
    left_padding = np.zeros((batch_size, 1))
    left_padded_reward = np.concatenate([left_padding, reward_np_array], axis=1)
    return np.diff(left_padded_reward, n=1, axis=1)


def phrase_reward_to_stepwise_reward(phrase_reward, eos_idx_mask):
    batch_size, seq_len = eos_idx_mask.size()
    stepwise_reward = np.zeros((batch_size, seq_len))
    for i in range(batch_size):
        pred_cnt = 0
        for j in range(seq_len):
            if eos_idx_mask[i, j].item() == 1:
                stepwise_reward[i, j] = phrase_reward[i, pred_cnt]
                pred_cnt += 1
            #elif j == seq_len:
            #    pass
    return stepwise_reward


def compute_pg_loss(log_likelihood, output_mask, q_val_sample):
    """
    :param log_likelihood: [batch_size, prediction_seq_len]
    :param input_mask: [batch_size, prediction_seq_len]
    :param q_val_sample: [batch_size, prediction_seq_len]
    :return:
    """
    log_likelihood = log_likelihood.view(-1)  # [batch_size * prediction_seq_len]
    output_mask = output_mask.view(-1)  # [batch_size * prediction_seq_len]
    q_val_sample = q_val_sample.view(-1)  # [batch_size * prediction_seq_len]
    objective = -log_likelihood * output_mask * q_val_sample
    objective = torch.sum(objective)/torch.sum(output_mask)
    return objective


if __name__ == "__main__":
    #reward = np.array([[1,3,5,6],[2,3,5,9]])
    #print(shape_reward(reward))

    #pred_word_2d_list = [['multi', 'agent', 'system'], ['agent', 'warning'], ['multi', 'agent'], ['agent'], ['agent', 'system'], ['multi', 'system'], ['what', 'is']]
    #trg_word_2d_list = [['multi', 'agent', 'system'], ['multi'], ['what', 'is']]
    #print(compute_match_result_new(trg_word_2d_list, pred_word_2d_list, type='exact'))
    #print(compute_match_result_new(trg_word_2d_list, pred_word_2d_list, type='sub'))
    #print(compute_match_result_new(trg_word_2d_list, pred_word_2d_list, type='exact', dimension=2))
    #print(compute_match_result_new(trg_word_2d_list, pred_word_2d_list, type='sub', dimension=2))

    #r = np.array([2, 1, 2, 0])
    #print(ndcg_at_k(r, 4, method=1))  # 0.96519546960144276

    r_2d = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0, 0, 1, 1, 0, 1, 0, 0]])
    k_list = [1,2,3]
    print(alpha_ndcg_at_ks(r_2d, k_list))
    r_2d = r_2d[:, np.array([0, 4, 6, 1, 5, 2, 7, 8, 9])]
    print(alpha_ndcg_at_ks(r_2d, k_list))

    '''
    r = np.array([0,1,1,0,1,0])
    k_list = [4, 6]
    print(average_precision_at_ks(r, k_list, num_trgs=5, num_predictions=6))
    '''
    pass
