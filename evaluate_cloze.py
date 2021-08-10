from summaqa.summaqa import evaluate_corpus, evaluate_corpus_batch
import re
from os.path import join, exists
import json
import os
import argparse
import torch
import time
from utils.cloze_model import ClozeModel
import pickle as pkl


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


def _read_file(filename):
    # print(dec_fname)
    summary_sent_list = []
    with open(filename) as f:
        for _, l in enumerate(f):
            summary_sent_list.append(l.strip())
    summary_str = ' '.join(summary_sent_list)
    return summary_str


def _construct_list(dec_dir, cloze_dir, cloze_model, is_non_numerical):
    print(dec_dir)
    print(cloze_dir)
    n_data = _count_data(cloze_dir)
    #n_data = 2
    masked_questions_ids_2dlist, answer_str_2dlist, summary_str_list = [], [], []
    for i in range(n_data):
        with open(join(cloze_dir, '{}.json'.format(i))) as f:
            js = json.loads(f.read())
        masked_question_list = js["masked_question_list"]
        if len(masked_question_list) > 0:
            answer_str_list = js["answer_list"]
            if is_non_numerical:
                first_400_non_numerical_question_indices = js["first_400_non_datetime_numerical_question_indices"]
                if len(first_400_non_numerical_question_indices) == 0:
                    continue
                masked_question_list = [masked_question_list[idx] for idx in
                                            first_400_non_numerical_question_indices]
                answer_str_list = [answer_str_list[idx] for idx in first_400_non_numerical_question_indices]

            masked_question_ids_list, answer_str_list_filtered = cloze_model.encode(masked_question_list, answer_str_list)
            if len(masked_question_ids_list) == 0:
                continue
            masked_questions_ids_2dlist.append(masked_question_ids_list)
            answer_str_list = answer_str_list_filtered
            answer_str_2dlist.append(answer_str_list)
            # read dec
            dec_fname = join(dec_dir, '{}.dec'.format(i))
            output_summary_str_lower = _read_file_lower(dec_fname)
            summary_str_list.append(output_summary_str_lower)
    assert len(summary_str_list) == len(masked_questions_ids_2dlist)
    return masked_questions_ids_2dlist, answer_str_2dlist, summary_str_list


def main(decode_dir, data, is_non_numerical):
    start_time = time.time()
    dec_dir = join(decode_dir, 'output')
    with open(join(decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    cloze_dir = join(data, 'cloze_entity_squad_with_idx_and_unanswerable_and_paraphrase_and_repeat', split)
    device = "cuda:0"
    cloze_model = ClozeModel(device)
    masked_questions_ids_2dlist, answer_str_2dlist, summary_str_list = _construct_list(dec_dir, cloze_dir, cloze_model, is_non_numerical)
    num_evaluation_samples = len(masked_questions_ids_2dlist)
    num_questions_per_sample = [len(questions) for questions in masked_questions_ids_2dlist]

    flattened_masked_questions_ids_list = []
    flattened_answer_str_list = []
    flattened_context_str_list = []
    for masked_question_ids_list, answer_str_list, summary_str in zip(masked_questions_ids_2dlist, answer_str_2dlist, summary_str_list):
        flattened_masked_questions_ids_list += masked_question_ids_list
        flattened_answer_str_list += answer_str_list
        flattened_context_str_list += [summary_str] * len(masked_question_ids_list)
    f1_score = cloze_model.compute_f1_score(flattened_masked_questions_ids_list, flattened_answer_str_list, flattened_context_str_list)
    assert f1_score.size(0) == len(flattened_masked_questions_ids_list)
    #print(f1_score)
    # confidence_score: [len(flattened_masked_questions_ids_list)]
    # compute average confidence for each sample
    num_processed_samples = 0
    score_for_each_sample = []
    for i in range(len(num_questions_per_sample)):
        # average for each batch
        if num_questions_per_sample[i] > 0:
            avg_score = f1_score[
                        num_processed_samples:num_processed_samples + num_questions_per_sample[i]].mean(dim=0)
            score_for_each_sample.append(avg_score)
        else:
            raise ValueError
        num_processed_samples += num_questions_per_sample[i]
    score_for_each_sample = torch.stack(score_for_each_sample, dim=0)  # [num_evaluation_samples]
    assert score_for_each_sample.size(0) == num_evaluation_samples
    avg_score = score_for_each_sample.mean(dim=0) # [1]
    #print(avg_score.size())
    print("Average f1 score: {:.5}".format(avg_score.item()))
    print("Processing time: {}s".format(time.time()-start_time))
    score_output_path = join(decode_dir, "qa_f1.pkl")
    with open(score_output_path, 'wb') as f:
        pkl.dump(score_for_each_sample.tolist(), f, pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Evaluate Summa_qa')
    )
    parser.add_argument('--decode_dir', action='store', required=True, help='directory of decoded summaries')
    parser.add_argument('--data', action='store', required=True, help='directory of decoded summaries')
    parser.add_argument('--is_non_numerical', action='store_true', help='directory of decoded summaries')
    args = parser.parse_args()
    main(args.decode_dir, args.data, args.is_non_numerical)
