import torch
import argparse
import config
import logging
import os
from os.path import join
import json
from utils import io
from utils.io import Many2ManyDatasetWithAttributes
from model.seq2seq import Seq2SeqModel
from model.seq2seq_style_input import Seq2SeqModelStyleInput
from model.seq2seq_exact_length_input import Seq2SeqModelExactLenInput
from model.diversity_attn_seq2seq import Seq2SeqDiversityAttnModel
from torch.utils.data import DataLoader
import pickle as pkl

import ml_pipeline

from utils.time_log import time_since
import datetime
import time
import numpy as np
import random


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    """
    opt.exp += '.ml'

    if opt.copy_attention:
        opt.exp += '.copy'

    if opt.coverage_attn:
        opt.exp += '.coverage'

    if opt.review_attn:
        opt.exp += '.review'

    if opt.orthogonal_loss:
        opt.exp += '.orthogonal'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'
    """

    if 2 in opt.control_modes:
        opt.model_type = 'seq2seq_exact_length_input'
    if 8 in opt.control_modes:
        opt.model_type = "seq2seq_diversity_attn"

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
        os.makedirs(join(opt.model_path, 'ckpt'))

    logging.info('EXP_PATH : ' + opt.exp_path)

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(join(opt.model_path, 'initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(join(opt.model_path, 'initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(join(opt.model_path, 'initial.json'), 'w'))

    return opt


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    if opt.copy_attention:
        logging.info('Training a seq2seq model with copy mechanism')
    else:
        logging.info('Training a seq2seq model')

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

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        # TODO: load the saved model and override the current one
    return model.to(opt.device)


def build_loader(data_path, batch_size, word2idx, src_max_len, trg_max_len, num_workers, opt):
    if opt.multi_style:
        style_label_map = {label : i for i, label in enumerate(opt.styles)}
        # dump style_label_map
        with open(join(opt.model_path, 'style_label_map.pkl'), 'wb') as f:
            pkl.dump(style_label_map, f, pkl.HIGHEST_PROTOCOL)
    else:
        style_label_map = None

    coll_fn_customized = io.coll_fn_with_attribute(word2idx=word2idx, style_label_map=style_label_map, src_max_len=src_max_len,
                                               trg_max_len=trg_max_len, control_modes=opt.control_modes, with_ground_truth=True)
    train_loader = DataLoader(Many2ManyDatasetWithAttributes('train', data_path, opt.control_modes), collate_fn=coll_fn_customized,
                              num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(Many2ManyDatasetWithAttributes('val', data_path, opt.control_modes), collate_fn=coll_fn_customized,
                              num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=False)

    #coll_fn_customized = io.coll_fn(word2idx=word2idx, src_max_len=src_max_len, trg_max_len=trg_max_len)
    #train_loader = DataLoader(Many2ManyDataset('train', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
    #                          batch_size=batch_size, pin_memory=True, shuffle=True)
    #valid_loader = DataLoader(Many2ManyDataset('val', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
    #                          batch_size=batch_size, pin_memory=True, shuffle=False)
    return train_loader, valid_loader


def main(opt):
    try:
        start_time = time.time()

        # construct vocab
        with open(join(opt.data, 'vocab_cnt.pkl'), 'rb') as f:
            wc = pkl.load(f)

        word2idx, idx2word = io.make_vocab_with_special_token(wc, opt.v_size, opt.control_modes)
        if 7 in opt.control_modes or 8 in opt.control_modes:
            print(word2idx['<ent>'])
            print(word2idx['<ent_end>'])
        """
        if opt.control_mode == 0: # control nothing
            word2idx, idx2word = io.make_vocab(wc, opt.v_size)
        elif opt.control_mode == 1: # control length
            word2idx, idx2word = io.make_vocab_with_len_bin(wc, opt.v_size)
        elif opt.control_mode == 2:
            word2idx, idx2word = io.make_vocab_with_exact_len_token(wc, opt.v_size)
        """
        opt.word2idx = word2idx
        opt.idx2word = idx2word

        # dump word2idx
        with open(join(opt.model_path, 'vocab.pkl'), 'wb') as f:
            pkl.dump(word2idx, f, pkl.HIGHEST_PROTOCOL)

        # construct loader
        load_data_time = time_since(start_time)
        train_data_loader, valid_data_loader = build_loader(opt.data, opt.batch_size, word2idx, opt.src_max_len, opt.trg_max_len, opt.batch_workers, opt)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        # construct model
        start_time = time.time()
        model = init_model(opt)
        if opt.w2v:
            # NOTE: the pretrained embedding having the same dimension
            #       as args.emb_dim should already be trained
            embedding, _ = io.make_embedding(idx2word, opt.w2v)
            model.set_embedding(embedding)
        logging.info(model)

        # construct optimizer
        optimizer_ml = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)

        # train the model
        ml_pipeline.train_model(model, optimizer_ml, train_data_loader, valid_data_loader, opt)

        training_time = time_since(start_time)
        logging.info('Model path: {}'.format(opt.model_path))
        logging.info('Time for training: {}'.format(datetime.timedelta(seconds=training_time)))

    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train_ml.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_opts(parser)
    config.train_ml_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False
    if 2 in opt.control_modes and 1 in opt.control_modes:  # cannot control length and length bin simultaneously
        raise ValueError
    if 3 in opt.control_modes and 4 in opt.control_modes:
        raise ValueError

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=False)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
