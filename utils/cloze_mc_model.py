from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import torch
import os
import json
import logging


MAX_CONTEXT_LEN = 120


class ClozeMCModel:
    def __init__(self, device):
        self.device = device
        self.model_path = "saved_models/entity_cloze_mc_coref_cnn_remove_duplicate"
        config = BertConfig.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, config=config, cache_dir="../../").cuda()
        self.model.eval()
        self.batch_size = 42
        ids = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[MC]'])
        self.cls_id = [ids[0]]
        self.sep_id = [ids[1]]
        self.mc_id = [ids[2]]

    def encode(self, masked_question_list, multiple_choices_list, answer_idx_list):
        masked_question_ids_list = []
        multiple_choices_ids_list = []
        answer_idx_list_filtered = []
        for masked_question, multiple_choices, answer_idx in zip(masked_question_list, multiple_choices_list,
                                                                 answer_idx_list):
            masked_question_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(masked_question))
            multiple_choices_str = multiple_choices[0] + ' [MC] ' + multiple_choices[1] + ' [MC] ' + multiple_choices[
                2] + ' [MC] ' + multiple_choices[3]
            multiple_choices_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(multiple_choices_str))
            if len(masked_question_ids) <= 125 and len(multiple_choices_ids) <= 125:
                masked_question_ids_list.append(masked_question_ids)
                multiple_choices_ids_list.append(multiple_choices_ids)
                answer_idx_list_filtered.append(answer_idx)

        return masked_question_ids_list, multiple_choices_ids_list, answer_idx_list_filtered

    def compute_batch(self, masked_questions_ids_batch, multiple_choices_ids_batch, answer_ids_batch, context_str_batch):
        input_ids_list = []
        input_lens = []
        token_type_ids_list = []
        for masked_question_ids, multiple_choices_ids, context_str in zip(masked_questions_ids_batch, multiple_choices_ids_batch, context_str_batch):
            context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context_str))
            context_ids = context_ids[:MAX_CONTEXT_LEN]
            first_segment_ids = self.cls_id + masked_question_ids + self.sep_id
            first_segment_end_position = len(first_segment_ids) - 1
            input_ids = first_segment_ids + context_ids + self.sep_id + multiple_choices_ids + self.sep_id
            input_ids_list.append(input_ids)
            input_lens.append(len(input_ids))
            token_type_ids = [0 if i <= first_segment_end_position else 1 for i in range( len(input_ids) )]
            token_type_ids_list.append(token_type_ids)
        max_input_len = max(input_lens)
        input_ids_list_padded = []
        token_type_ids_list_padded = []
        for input_ids, token_type_ids in zip(input_ids_list, token_type_ids_list):
            padding_len = max_input_len - len(input_ids)
            input_ids_list_padded.append(input_ids + [self.tokenizer.pad_token_id] * padding_len)
            token_type_ids_list_padded.append(token_type_ids + [1] * padding_len)
        with torch.no_grad():
            input_ids_tensor = torch.LongTensor(input_ids_list_padded).to(self.device)
            token_type_ids_tensor = torch.LongTensor(token_type_ids_list_padded).to(self.device)
            answer_ids_tensor = torch.LongTensor(answer_ids_batch).to(self.device)
            attention_mask = torch.ne(input_ids_tensor, self.tokenizer.pad_token_id).float()
            try:
                outputs = self.model(input_ids=input_ids_tensor, attention_mask=attention_mask, token_type_ids=token_type_ids_tensor)
            except:
                logging.info(input_ids_tensor.size())
                logging.info(attention_mask.size())
                logging.info(token_type_ids_tensor.size())
                logging.info()
                for input_ids in input_ids_list_padded:
                    logging.info(input_ids)
                    logging.info(self.tokenizer(input_ids, clean_up_tokenization_spaces=False))
                for token_type_ids in token_type_ids_list_padded:
                    logging.info(token_type_ids)
                logging.info(attention_mask.cpu().numpy())
                exit()
            logits = outputs[0]  # [batch_size, 4]
            confidence_scores_all = torch.nn.functional.softmax(logits, dim=1)  # [batch_size, 4]
            ground_truth_confidence_scores = torch.gather(confidence_scores_all, dim=1, index=answer_ids_tensor.unsqueeze(1))  # [batch_size]
        return ground_truth_confidence_scores

    def compute_confidence_score(self, masked_questions_ids_list, multiple_choices_ids_list, answer_id_list, context_str_list):
        iter_range = range(0, len(masked_questions_ids_list), self.batch_size)
        confidence_scores_list = []
        for batch_start in iter_range:
            masked_questions_ids_batch = masked_questions_ids_list[batch_start:batch_start+self.batch_size]
            multiple_choices_ids_batch = multiple_choices_ids_list[batch_start:batch_start+self.batch_size]
            answer_id_batch = answer_id_list[batch_start:batch_start+self.batch_size]
            context_str_batch = context_str_list[batch_start:batch_start+self.batch_size]
            confidence_scores = self.compute_batch(masked_questions_ids_batch, multiple_choices_ids_batch, answer_id_batch, context_str_batch)
            # [batch_size]
            confidence_scores_list.append(confidence_scores)
        return torch.cat(confidence_scores_list, dim=0)  # [len(masked_questions_ids_list)]


if __name__ == "__main__":
    cloze_model = ClozeMCModel("cuda:0")
    split_dir="/mnt/sharedfolder/hpchan/datasets/cased-cnn-dailymail_coref/train"
    with open(os.path.join(split_dir, "{}.json".format(0))) as f:
        js = json.loads(f.read())
    """
    masked_questions_ids_list = js["masked_question_ids_list"]
    multiple_choices_ids_list = js["multiple_choices_ids_list"]
    answer_id_list = js["answer_idx_list"]
    summary_sent_list = js["abstract"]
    summary_str = ' '.join(summary_sent_list)
    context_str_list = [summary_str for i in range(len(answer_id_list))]
    context_str_list[-1] = "Mentally ill inmates in Miami are housed on the `` forgotten floor '' Judge Steven Leifman says most are there as a result of `` avoidable felonies ''"
    confidence_score = cloze_model.compute_confidence_score(masked_questions_ids_list, multiple_choices_ids_list, answer_id_list, context_str_list)
    print(masked_questions_ids_list)
    print(multiple_choices_ids_list)
    print(context_str_list)
    print(answer_id_list)
    print(len(answer_id_list))
    print(confidence_score)
    
    """
    masked_questions_ids_2dlist = []
    multiple_choices_ids_2dlist = []
    answer_id_2dlist = []
    context_str_2dlist = []
    for i in range(3):
        with open(os.path.join(split_dir, "{}.json".format(i))) as f:
            js = json.loads(f.read())
        masked_questions_ids_list = js["masked_question_ids_list"]
        multiple_choices_ids_list = js["multiple_choices_ids_list"]
        answer_id_list = js["answer_idx_list"]
        summary_sent_list = js["abstract"]
        summary_str = ' '.join(summary_sent_list)
        context_str_list = [summary_str for i in range(len(answer_id_list))]
        masked_questions_ids_2dlist.append(masked_questions_ids_list)
        multiple_choices_ids_2dlist.append(multiple_choices_ids_list)
        answer_id_2dlist.append(answer_id_list)
        context_str_2dlist.append(context_str_list)

    num_questions_per_sample = [len(questions) for questions in masked_questions_ids_2dlist]
    print(num_questions_per_sample)
    flattened_masked_questions_ids_list = []
    flattened_answer_id_list = []
    flattened_multiple_choices_ids_list = []
    flattened_context_str_list = []
    for masked_question_ids_list, answer_id_list, multiple_choices_ids_list, context_str_list in zip(masked_questions_ids_2dlist, answer_id_2dlist,
                                                                     multiple_choices_ids_2dlist, context_str_2dlist):
        flattened_masked_questions_ids_list += masked_question_ids_list
        flattened_answer_id_list += answer_id_list
        flattened_multiple_choices_ids_list += multiple_choices_ids_list
        flattened_context_str_list += context_str_list

    print(flattened_context_str_list)
    print(len(flattened_context_str_list))
    print(len(flattened_masked_questions_ids_list))
    print(len(flattened_answer_id_list))
    print(len(flattened_multiple_choices_ids_list))
    confidence_score = cloze_model.compute_confidence_score(flattened_masked_questions_ids_list, flattened_multiple_choices_ids_list, flattened_answer_id_list, flattened_context_str_list)
    print(confidence_score)
    num_processed_samples = 0
    score_for_each_batch = []
    for i in range(len(num_questions_per_sample)):
        # average for each batch
        avg_score = confidence_score[num_processed_samples:num_processed_samples+num_questions_per_sample[i]].mean(dim=0)
        score_for_each_batch.append(avg_score)
        print(num_processed_samples)
        print(num_processed_samples+num_questions_per_sample[i])
        num_processed_samples += num_questions_per_sample[i]
    print(score_for_each_batch)
    score_for_each_batch = torch.cat(score_for_each_batch, dim=0)
    print(score_for_each_batch)
    print(score_for_each_batch.size())
