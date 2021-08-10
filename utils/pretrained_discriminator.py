import torch
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from utils.masked_softmax import MaskedSoftmax
from utils.utils_glue import InputExample
from utils.utils_glue import convert_examples_to_tensors_for_bert_seq_classify


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}

MAX_CAND_SEQ_LEN = 122

class SeqClassifyDiscriminator:
    def __init__(self, model_dir, ckpt_dir, device):
        self.softmax = MaskedSoftmax(dim=1)
        self.model_type = "bert"
        #self.model_name_or_path = "bert-base-uncased"
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        self.pretrained_model = model_class.from_pretrained(ckpt_dir)
        self.pretrained_model.to(device)
        self.pretrained_model.eval()
        self.tokenizer = tokenizer_class.from_pretrained(model_dir, do_lower_case=True)
        self.device = device

    def score(self, cands, refs=None):
        """
        :param cands: a list of str
        :param refs: a list of str
        :return:
        """
        example_list = []
        for cand_str in cands:
            example_list.append(InputExample(guid=None, text_a=cand_str, text_b=None, label=None) )
        batch = convert_examples_to_tensors_for_bert_seq_classify(example_list, MAX_CAND_SEQ_LEN, self.tokenizer,
                                                    cls_token_at_end=bool(self.model_type in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=self.tokenizer.cls_token,
                                                    sep_token=self.tokenizer.sep_token,
                                                    cls_token_segment_id=2 if self.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(self.model_type in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0)

        # move data to GPU if available
        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None
                      # XLM don't use segment_ids
                      }
            outputs = self.pretrained_model(**inputs)
            logits = outputs[0]  # [batch, num_classes]
            class_prob = self.softmax(logits, mask=None)  # [batch, num_classes]
            return class_prob

