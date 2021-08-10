import os
import pickle
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import json
import re
from os.path import join

def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data_dir, split, logger):
        split_dir = os.path.join(data_dir, split)
        cached_features_file = os.path.join(data_dir, args.model_type + '_cached_lm_' + split)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", data_dir)

            self.examples = []

            n_data = _count_data(split_dir)

            for i in range(n_data):
                js = json.load(open(join(split_dir, '{}.json'.format(i))))
                if js['article'] and js['abstract']:
                    doc_sent_list = js['article']
                    doc_str = ' '.join(doc_sent_list)
                    #doc_word_list = doc_str.split(' ')
                    #doc_str_truncated  = ' '.join(doc_word_list[:MAX_DOC_LEN])

                    summary_sent_list = js['abstract']
                    summary_str = ' '.join(summary_sent_list)
                    #summary_word_list = summary_str.split(' ')
                    #summary_str_truncated = ' '.join(summary_word_list[:MAX_SUMM_LEN])

                    #doc_str_truncated = doc_str_truncated + " TL;DR:"
                    doc_str_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc_str))
                    #doc_str_tokenized = doc_str_tokenized[:MAX_DOC_LEN+4]  # account for TL;DR:
                    summary_str_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(summary_str))
                    #summary_str_tokenized = summary_str_tokenized[:MAX_SUMM_LEN]

                    #text = doc_str_truncated + " TL;DR: " + summary_str_truncated
                    #tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                    self.examples.append( (tokenizer.build_inputs_with_special_tokens(doc_str_tokenized), tokenizer.build_inputs_with_special_tokens(summary_str_tokenized), doc_sent_list, summary_sent_list) )

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
