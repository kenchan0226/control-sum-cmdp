import torch
import config
import argparse
import pickle as pkl
from utils import io
from utils.io import Many2ManyDatasetWithAttributes
from torch.utils.data import DataLoader
import os
from os.path import join
from model.seq2seq import Seq2SeqModel
from model.seq2seq_style_input import Seq2SeqModelStyleInput
from model.seq2seq_exact_length_input import Seq2SeqModelExactLenInput
from model.diversity_attn_seq2seq import Seq2SeqDiversityAttnModel
from sequence_generator import SequenceGenerator
from tqdm import tqdm
import json
from utils.string_helper import prediction_to_sentence
import nltk
import rreplace
from types import SimpleNamespace


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    # make directory
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
        os.makedirs(join(opt.pred_path, 'output'))
    else:
        print("Folder exists!")
        raise ValueError

    # dump configuration
    torch.save(opt, open(join(opt.pred_path, 'decode.config'), 'wb'))
    json.dump(vars(opt), open(join(opt.pred_path, 'log.json'), 'w'))

    return opt


def init_pretrained_model(pretrained_model_path, opt):
    if opt.model_type == 'seq2seq':
        assert not opt.multi_style
        model = Seq2SeqModel(opt)
    elif opt.model_type == 'seq2seq_style_input':
        assert opt.multi_style
        model = Seq2SeqModelStyleInput(opt)
    elif opt.model_type == 'seq2seq_exact_length_input':
        model = Seq2SeqModelExactLenInput(opt)
    elif opt.model_type == 'seq2seq_diversity_attn':
        model = Seq2SeqDiversityAttnModel(opt)
    else:
        raise ValueError

    model.load_state_dict(torch.load(pretrained_model_path))
    model.to(opt.device)
    model.eval()
    return model


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk, src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, oov, src_word_list in zip(predictions, scores, attention, oov_lists, src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, attn)
            #sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict['sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict['attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def predict(test_data_loader, model, opt):
    generator = SequenceGenerator(model,
                                  bos_idx=io.BOS,
                                  eos_idx=io.EOS,
                                  pad_idx=io.PAD,
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.pred_max_len,
                                  include_attn_dist=opt.include_attn_dist,
                                  length_penalty_factor=opt.length_penalty_factor,
                                  coverage_penalty_factor=opt.coverage_penalty_factor,
                                  length_penalty=opt.length_penalty,
                                  coverage_penalty=opt.coverage_penalty,
                                  cuda=opt.gpuid > -1,
                                  n_best=opt.n_best,
                                  block_ngram_repeat=opt.block_ngram_repeat,
                                  ignore_when_blocking=opt.ignore_when_blocking,
                                  len_idx=opt.word2idx[io.EXACT_LEN_WORD] if 2 in opt.control_modes else -1
                                  )

    num_exported_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            #src, src_lens, src_mask, src_oov, oov_lists, src_str_list, original_idx_list = batch
            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
            oov_lists: a list of oov words for each src, 2dlist
            """
            src = batch['src_tensor']
            src_lens = batch['src_lens']
            src_mask = batch['src_mask']
            src_oov = batch['src_oov_tensor']
            oov_lists = batch['oov_lists']
            src_str_list = batch['src_list_tokenized']
            #original_idx_list = batch['original_indices']

            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)

            """
            for src_str in src_str_list:
                print(src_str[:10])
            print(src.detach().cpu().numpy()[:, :10])
            print(batch['trg_lens'])
            print(batch['len_bins'])
            print(batch['exact_lens'])
            exit()
            """

            if opt.multi_style:
                style_label = batch['style_tensor']
                style_label = style_label.to(opt.device)

            if isinstance(model, Seq2SeqModel):
                beam_search_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, opt.word2idx)
            elif isinstance(model, Seq2SeqModelStyleInput):
                beam_search_result = generator.beam_search_with_style(src, src_lens, src_oov, src_mask, oov_lists,
                                                                      opt.word2idx, style_label)
            elif isinstance(model, Seq2SeqModelExactLenInput):
                beam_search_result = generator.beam_search_with_exact_len(src, src_lens, src_oov, src_mask, oov_lists,
                                                                          opt.word2idx, batch['exact_lens'])
            elif isinstance(model, Seq2SeqDiversityAttnModel):
                query_tensor = batch['query_tensor'].to(opt.device)
                query_mask = batch['query_mask'].to(opt.device)
                query_lens = batch['query_lens']
                beam_search_result = generator.beam_search_diversity_attn(src, src_lens, query_tensor, query_lens, src_oov, src_mask, query_mask, oov_lists, opt.word2idx)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists, io.EOS, io.UNK, opt.replace_unk, src_str_list)
            # list of {"sentences": [], "scores": [], "attention": []}

            # recover the original order in the dataset
            #seq_pairs = sorted(zip(original_idx_list, src_str_list, pred_list, oov_lists), key=lambda p: p[0])
            #original_idx_list, src_str_list, pred_list, oov_lists = zip(*seq_pairs)

            # Process every src in the batch

            for src_str, pred, oov in zip(src_str_list, pred_list, oov_lists):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words
                pred_str_list = pred['sentences']  # predicted sentences from a single src, a list of list of word, with len=[n_best, out_seq_len], does not include the final <EOS>
                pred_score_list = pred['scores']
                pred_attn_list = pred['attention']  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]

                decode_out_str = ' '.join(pred_str_list[0])
                decode_out_sent_list = nltk.tokenize.sent_tokenize(decode_out_str)

                # output the predicted sentences to a file
                with open(join(opt.pred_path, 'output/{}.dec'.format(num_exported_samples)), 'w') as f:
                    f.write(io.make_html_safe('\n'.join(decode_out_sent_list)))
                num_exported_samples += 1


def main(opt):
    # load word2idx and idx2word
    model_dir_path = os.path.dirname(opt.pretrained_model)
    model_dir_path = rreplace.rreplace(model_dir_path, 'ckpt', '', 1)
    with open(join(model_dir_path, 'vocab.pkl'), 'rb') as f:
        word2idx = pkl.load(f)
    idx2word = {i: w for w, i in word2idx.items()}
    opt.word2idx = word2idx
    opt.idx2word = idx2word
    opt.vocab_size = len(word2idx)

    # load style label map
    if opt.multi_style:
        with open(join(model_dir_path, 'style_label_map.pkl'), 'rb') as f:
            style_label_map = pkl.load(f)
    else:
        style_label_map = None

    if opt.target_style != "":
        target_style_idx = style_label_map[opt.target_style]
    else:
        target_style_idx = -1

    # init the pretrained model
    #old_opt = torch.load(join(model_dir_path, "initial.config"))
    old_opt_dict = json.load(open(join(model_dir_path, "initial.json")))
    old_opt = SimpleNamespace(**old_opt_dict)
    old_opt.word2idx = word2idx
    old_opt.idx2word = idx2word
    old_opt.device = opt.device
    opt.control_modes = old_opt.control_modes

    if len(opt.control_modes) > 0:
        assert opt.with_ground_truth_input or len(opt.desired_target_numbers) == len(opt.control_modes)
    assert opt.multi_style == old_opt.multi_style
    model = init_pretrained_model(opt.pretrained_model, old_opt)

    coll_fn_customized = io.coll_fn_with_attribute(word2idx=word2idx, style_label_map=style_label_map,
                                               with_style=opt.with_groundtruth_style,
                                               target_style_idx=target_style_idx, src_max_len=opt.src_max_len,
                                               trg_max_len=-1,
                                               control_modes=opt.control_modes, with_ground_truth=opt.with_ground_truth_input,
                                               desired_target_numbers=opt.desired_target_numbers,
                                               is_multiple_ref=opt.multiple_reference)

    test_loader = DataLoader(Many2ManyDatasetWithAttributes(opt.split, opt.data, opt.control_modes),
                             collate_fn=coll_fn_customized,
                             num_workers=opt.batch_workers,
                             batch_size=opt.batch_size, pin_memory=True, shuffle=False)

    # Print out predict path
    print("Prediction path: %s" % opt.pred_path)

    # output the summaries to opt.pred_path/output
    predict(test_loader, model, opt)


if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.predict_opts(parser)
    opt = parser.parse_args()

    opt = process_opt(opt)

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    assert not (opt.with_ground_truth_input and len(opt.desired_target_numbers) > 0)
    assert not (opt.with_groundtruth_style and opt.target_style != "")

    main(opt)

