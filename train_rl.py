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
from model.lagrangian import Lagrangian
from torch.utils.data import DataLoader
import pickle as pkl
import rreplace
import rl_pipeline

from utils.time_log import time_since
import datetime
import time
import numpy as np
import random
from types import SimpleNamespace


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    #opt.exp += '.rl'

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
        os.makedirs(join(opt.model_path, 'ckpt'))
        os.makedirs(join(opt.model_path, 'epoch_states'))

    logging.info('EXP_PATH : ' + opt.exp_path)

    # dump the setting (opt) to disk in order to reuse easily
    """
    if opt.train_from:
        previous_opt = torch.load(
            open(join(opt.model_path, 'rl.config'), 'rb')
        )
        opt.pretrained_model = previous_opt
    """
    torch.save(opt,
               open(join(opt.model_path, 'rl.config'), 'wb')
               )
    json.dump(vars(opt), open(join(opt.model_path, 'rl.json'), 'w'))

    return opt


def init_pretrained_model(pretrained_state_dict, opt):
    if opt.model_type == 'seq2seq':
        assert not opt.multi_style
        model = Seq2SeqModel(opt)
    elif opt.model_type == 'seq2seq_style_input':
        assert opt.multi_style
        model = Seq2SeqModelStyleInput(opt)
    elif opt.model_type == 'seq2seq_exact_length_input':
        model = Seq2SeqModelExactLenInput(opt)
    else:
        raise ValueError
    model.load_state_dict(pretrained_state_dict)
    model.to(opt.device)
    model.eval()
    return model


def build_loader(data_path, batch_size, word2idx, src_max_len, trg_max_len, num_workers, ml_opt):
    if ml_opt.multi_style:
        style_label_map = {label : i for i, label in enumerate(opt.styles)}
        # dump style_label_map
        with open(join(opt.model_path, 'style_label_map.pkl'), 'wb') as f:
            pkl.dump(style_label_map, f, pkl.HIGHEST_PROTOCOL)
    else:
        style_label_map = None

    coll_fn_customized = io.coll_fn_with_attribute(word2idx=word2idx, style_label_map=style_label_map, src_max_len=src_max_len,
                                               trg_max_len=trg_max_len, control_modes=ml_opt.control_modes, with_ground_truth=True,
                                                   is_rl=True)
    print("loader")
    train_loader = DataLoader(Many2ManyDatasetWithAttributes('train', data_path, ml_opt.control_modes), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(Many2ManyDatasetWithAttributes('val', data_path, ml_opt.control_modes), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=False)
    print("finish")
    return train_loader, valid_loader


def main(opt):
    try:
        start_time = time.time()
        # load word2idx and idx2word
        ml_pretrained_model_dir_path = os.path.dirname(opt.pretrained_model)
        ml_pretrained_model_dir_path = rreplace.rreplace(ml_pretrained_model_dir_path, 'ckpt', '', 1)
        with open(join(ml_pretrained_model_dir_path, 'vocab.pkl'), 'rb') as f:
            word2idx = pkl.load(f)
        idx2word = {i: w for w, i in word2idx.items()}
        opt.word2idx = word2idx
        opt.idx2word = idx2word
        opt.vocab_size = len(word2idx)

        # dump word2idx
        with open(join(opt.model_path, 'vocab.pkl'), 'wb') as f:
            pkl.dump(word2idx, f, pkl.HIGHEST_PROTOCOL)

        # load the config of ml_pretrained model and dump the config to the rl model dir
        #old_opt = torch.load(join(ml_pretrained_model_dir_path, "initial.config"))
        ml_old_opt_dict = json.load(open(join(ml_pretrained_model_dir_path, "initial.json")))
        ml_old_opt = SimpleNamespace(**ml_old_opt_dict)
        json.dump(ml_old_opt_dict, open(join(opt.model_path, 'initial.json'), 'w'))
        torch.save(ml_old_opt, open(join(opt.model_path, 'initial.config'), 'wb'))

        # construct loader
        load_data_time = time_since(start_time)
        train_data_loader, valid_data_loader = build_loader(opt.data, opt.batch_size, word2idx, opt.src_max_len,
                                                            opt.trg_max_len, opt.batch_workers, ml_old_opt)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        # init the pretrained model
        ml_old_opt.word2idx = word2idx
        ml_old_opt.idx2word = idx2word
        ml_old_opt.device = opt.device
        opt.control_modes = ml_old_opt.control_modes
        if opt.train_from:
            epoch_state_dict = torch.load(opt.train_from)
            model = init_pretrained_model(epoch_state_dict['model'], ml_old_opt)
            optimizer_rl = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                            lr=opt.learning_rate)
            optimizer_rl.load_state_dict(epoch_state_dict['optimizer_rl'])
            opt.start_epoch = epoch_state_dict['epoch'] + 1
            if opt.constrained_mdp:
                lagrangian_model = Lagrangian(len(opt.cost_types), opt.cost_thresholds, opt.lagrangian_init_val, opt.use_lagrangian_hinge_loss)
                lagrangian_model.load_state_dict(epoch_state_dict['lagrangian_model'])
                lagrangian_model.to(opt.device)
                optimizer_lagrangian = torch.optim.Adam(params=filter(lambda p: p.requires_grad, lagrangian_model.parameters()), lr=opt.learning_rate_multiplier)
                optimizer_lagrangian.load_state_dict(epoch_state_dict['optimizer_lagrangian'])
                lagrangian_params = (lagrangian_model, optimizer_lagrangian)
            else:
                lagrangian_params = None
            epoch_state_dict['model'] = None
            epoch_state_dict['optimizer_rl'] = None
            epoch_state_dict['lagrangian_model'] = None
            epoch_state_dict['optimizer_lagrangian'] = None
        else:
            model = init_pretrained_model(torch.load(opt.pretrained_model), ml_old_opt)
            # construct optimizer
            optimizer_rl = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
            if opt.constrained_mdp:
                lagrangian_model = Lagrangian(len(opt.cost_types), opt.cost_thresholds, opt.lagrangian_init_val, opt.use_lagrangian_hinge_loss)
                lagrangian_model.to(opt.device)
                optimizer_lagrangian = torch.optim.Adam(params=filter(lambda p: p.requires_grad, lagrangian_model.parameters()), lr=opt.learning_rate_multiplier)
                lagrangian_params = (lagrangian_model, optimizer_lagrangian)
            else:
                lagrangian_params = None
            epoch_state_dict = None

        # train the model
        rl_pipeline.train_model(model, optimizer_rl, train_data_loader, valid_data_loader, opt, lagrangian_params, epoch_state_dict)

        training_time = time_since(start_time)
        logging.info('Model path: {}'.format(opt.model_path))
        logging.info('Time for training: {}'.format(datetime.timedelta(seconds=training_time)))
    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train_ml.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.train_rl_opts(parser)
    opt = parser.parse_args()
    #print("ml loss coef: {}".format(opt.ml_loss_coefficient))
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    #print("ml loss coef: {}".format(opt.ml_loss_coefficient))

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    assert len(opt.cost_types) == len(opt.cost_thresholds)

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=False)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
