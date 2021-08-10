 # code from huggingface see https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering

import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from utils.utils_glue import InputExample, convert_examples_to_tensors_for_bert_qa
from utils.masked_softmax import MaskedSoftmax
MAX_LEN = 256

class QA_Bert():
    def __init__(self, device='cuda:0'):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        #self.SEP_id = self.tokenizer.encode('[SEP]')[0]
        self.SEP_id = self.tokenizer.sep_token_id
        self.model_type = "bert"
        self.softmax = MaskedSoftmax(dim=1)
        self.device = device
        self.model.to(device)

    def predict_batch(self, question_list, evaluated_text):
        example_list = []
        """
        for question, text in zip(question_list, text_list):
            example_list.append(InputExample(guid=None, text_a=question, text_b=text, label=None))

        batch = convert_examples_to_tensors_for_bert_qa(example_list, 512, self.tokenizer,
                                                                  cls_token_at_end=bool(self.model_type in ['xlnet']),
                                                                  # xlnet has a cls token at the end
                                                                  cls_token=self.tokenizer.cls_token,
                                                                  sep_token=self.tokenizer.sep_token,
                                                                  cls_token_segment_id=2 if self.model_type in [
                                                                      'xlnet'] else 0,
                                                                  pad_on_left=bool(self.model_type in ['xlnet']),
                                                                  # pad on the left for xlnet
                                                                  pad_token_segment_id=4 if self.model_type in [
                                                                      'xlnet'] else 0)
        """
        input_ids_list = []
        token_type_ids_list = []
        max_input_len = 0
        evaluated_text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(evaluated_text))[:MAX_LEN]

        for question in question_list:
            question_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(question))[:MAX_LEN]
            input_ids = self.tokenizer.build_inputs_with_special_tokens(question_ids, evaluated_text_ids)
            input_ids_list.append(input_ids)
            #print(input_ids)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(question_ids, evaluated_text_ids)
            token_type_ids_list.append(token_type_ids)
            #print(token_type_ids)
            if len(input_ids) > max_input_len:
                max_input_len = len(input_ids)
        # padding
        input_ids_list_padded = []
        token_type_ids_list_padded = []
        for input_ids, token_type_ids in zip(input_ids_list, token_type_ids_list):
            padding_length = max_input_len - len(input_ids)
            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            input_ids_list_padded.append(input_ids)
            token_type_ids = token_type_ids + ([1] * padding_length)
            token_type_ids_list_padded.append(token_type_ids)
        input_ids_tensor = torch.LongTensor(input_ids_list_padded).to(self.device)
        token_type_ids_tensor = torch.LongTensor(token_type_ids_list_padded).to(self.device)
        # attention mask
        attention_mask = torch.ne(input_ids_tensor, self.tokenizer.pad_token_id).float()

        with torch.no_grad():
            inputs = {'input_ids': input_ids_tensor,
                      'attention_mask': attention_mask,
                      'token_type_ids': token_type_ids_tensor
                      }
            start_logits, end_logits = self.model(**inputs)
            # [batch, seq_len], [batch, seq_len]

        softmax_mask = attention_mask * token_type_ids_tensor  # [batch, seq_len]
        #print(softmax_mask)
        start_scores = self.softmax(start_logits, mask=softmax_mask)  # [batch, seq_len]
        end_scores = self.softmax(end_logits, mask=softmax_mask)  # [batch, seq_len]

        # start_values [batch, 1], start_indices [batch, 1]
        start_values, start_indices = start_scores.topk(k=1, dim=1)
        # end_values [batch, 1], end_indices [batch, 1]
        end_values, end_indices = end_scores.topk(k=1, dim=1)

        asw_list = []
        for i, input_ids in enumerate(input_ids_list):
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            asw = ' '.join(input_tokens[start_indices[i][0]: end_indices[i][0] + 1])
            asw_list.append(asw)

        probs = start_values[:, 0] * end_values[:, 0]  # [batch]
        #prob_sum = probs.sum()  # [1]

        #print(asw_list)
        #print(probs)
        #exit()

        return asw_list, probs

    def predict(self, question, text):

        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = self.tokenizer.encode(input_text)  # a list of int
        token_type_ids = [0 if i <= input_ids.index(self.SEP_id) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.model(torch.tensor([input_ids]).to(self.device), )
        #print(start_scores)
        #print(end_scores)
        #print()
        start_scores = torch.functional.F.softmax(start_scores, -1) * torch.Tensor(token_type_ids).to(self.device)
        end_scores = torch.functional.F.softmax(end_scores, -1) * torch.Tensor(token_type_ids).to(self.device)

        start_values, start_indices = start_scores.topk(1)
        end_values, end_indices = end_scores.topk(1)

        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        asw = ' '.join(all_tokens[start_indices[0][0] : end_indices[0][0]+1])
        prob = start_values[0][0] * end_values[0][0]

        #print("asw: {}".format(asw))
        #print("prob : {}".format(prob.item()))

        return asw, prob.item()