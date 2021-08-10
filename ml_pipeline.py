import torch.nn as nn
from utils.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.time_log import time_since
from validation import evaluate_loss
import time
import math
import logging
import torch
import sys
import os
from utils.report import export_train_and_valid_loss
from utils.io import remove_old_ckpts
from utils.io import LEN_WORD, LEN_BINS

EPS = 1e-8

def train_model(model, optimizer_ml, train_data_loader, valid_data_loader, opt):
    '''
    generator = SequenceGenerator(model,
                                  eos_idx=opt.word2idx[io.EOS_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length
                                  )
    '''
    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    previous_valid_loss = float('inf')
    num_stop_dropping = 0


    if opt.train_from:  # opt.train_from:
        #TODO: load the training state
        raise ValueError("Not implemented the function of load from trained model")
        pass

    model.train()

    for epoch in range(opt.start_epoch, opt.epochs+1):
        if early_stop_flag:
            break

        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            # Training
            batch_loss_stat, decoder_dist = train_one_batch(batch, model, optimizer_ml, opt, batch_i)
            report_train_loss_statistics.update(batch_loss_stat)
            total_train_loss_statistics.update(batch_loss_stat)
            #logging.info("one_batch")
            #report_loss.append(('train_ml_loss', loss_ml))
            #report_loss.append(('PPL', loss_ml))

            # Brief report
            '''
            if batch_i % opt.report_every == 0:
                brief_report(epoch, batch_i, one2one_batch, loss_ml, decoder_log_probs, opt)
            '''

            if total_batch % opt.checkpoint_interval == 0:
                print("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            # Save the model parameters if the validation loss improved.
            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):
                    print("Enter check point!")
                    sys.stdout.flush()
                    # log training loss and training ppl
                    current_train_ppl = report_train_loss_statistics.ppl()
                    current_train_loss = report_train_loss_statistics.xent()
                    report_train_ppl.append(current_train_ppl)
                    report_train_loss.append(current_train_loss)
                    # Run validation and log valid loss and ppl
                    valid_loss_stat = evaluate_loss(valid_data_loader, model, opt)
                    model.train()
                    current_valid_loss = valid_loss_stat.xent()
                    current_valid_ppl = valid_loss_stat.ppl()
                    report_valid_ppl.append(current_valid_ppl)
                    report_valid_loss.append(current_valid_loss)
                    # debug
                    if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
                        exit()
                    # print out train and valid loss
                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'avg training ppl: %.3f; avg validation ppl: %.5f; best validation ppl: %.5f' % (
                            current_train_ppl, current_valid_ppl, best_valid_ppl))
                    logging.info(
                        'avg training loss: %.3f; avg validation loss: %.5f; best validation loss: %.5f' % (
                            current_train_loss, current_valid_loss, best_valid_loss))

                    if epoch >= opt.start_decay_and_early_stop_at:
                        if current_valid_loss < previous_valid_loss: # update the best valid loss and save the model parameters
                            print("Valid loss drops")
                            sys.stdout.flush()
                            if current_valid_loss < best_valid_loss:
                                best_valid_loss = current_valid_loss
                                best_valid_ppl = current_valid_ppl
                            num_stop_dropping = 0

                            check_pt_model_path = os.path.join(opt.model_path, 'ckpt', '%s-epoch-%d-total_batch-%d-valid_ppl-%.5f' % (
                                opt.exp, epoch, total_batch, current_valid_ppl))
                            torch.save(  # save model parameters
                                model.state_dict(),
                                open(check_pt_model_path, 'wb')
                            )
                            logging.info('Saving checkpoint to %s' % check_pt_model_path)

                            # Only keep the highest three checkpoints
                            remove_old_ckpts(opt.model_path, reverse=False)
                        else:
                            print("Valid loss does not drop")
                            sys.stdout.flush()
                            num_stop_dropping += 1
                            # decay the learning rate by a factor
                            if opt.learning_rate_decay < 1:
                                for i, param_group in enumerate(optimizer_ml.param_groups):
                                    old_lr = float(param_group['lr'])
                                    new_lr = old_lr * opt.learning_rate_decay
                                    if new_lr < opt.min_lr:
                                        new_lr = opt.min_lr
                                    if old_lr - new_lr > EPS:
                                        param_group['lr'] = new_lr
                                logging.info('Learning rate drops to {}'.format(new_lr))

                        previous_valid_loss = current_valid_loss

                        if not opt.disable_early_stop:
                            if num_stop_dropping >= opt.early_stop_tolerance:
                                logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)
                                early_stop_flag = True
                                break

                    report_train_loss_statistics.clear()


    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl, opt.checkpoint_interval, train_valid_curve_path)

    # log best loss
    logging.info("final_best_valid_loss: %.3f" % best_valid_loss)
    logging.info("final_best_valid_ppl: %.3f" % best_valid_ppl)


def train_one_batch(batch, model, optimizer, opt, batch_i):
    #src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_sent_2d_list, trg, trg_oov, trg_lens, trg_mask, _ = batch
    """
    trg: LongTensor [batch, trg_seq_len], each target trg[i] contains the indices of a set of concatenated keyphrases, separated by opt.word2idx[io.SEP_WORD]
                 if opt.delimiter_type = 0, SEP_WORD=<sep>, if opt.delimiter_type = 1, SEP_WORD=<eos>
    trg_oov: same as trg_oov, but all unk words are replaced with temporary idx, e.g. 50000, 50001 etc.
    """
    src = batch['src_tensor']
    src_lens = batch['src_lens']
    src_mask = batch['src_mask']
    src_oov = batch['src_oov_tensor']
    oov_lists = batch['oov_lists']
    src_str_list = batch['src_list_tokenized']
    trg_sent_2d_list = batch['trg_sent_2d_list']
    trg = batch['trg_tensor']
    trg_oov = batch['trg_oov_tensor']
    trg_lens = batch['trg_lens']
    trg_mask = batch['trg_mask']
    position_ids = batch['position_ids']
    if opt.multi_style:
        style_label = batch['style_tensor']  # [batch_size]
        style_label = style_label.to(opt.device)

    batch_size = src.size(0)
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)
    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)
    if position_ids is not None:
        position_ids = position_ids.to(opt.device)

    optimizer.zero_grad()

    # debug
    """
    for src_str in src_str_list:
        print(src_str[:10])
    print(src.detach().cpu().numpy()[:, :10])
    print(batch['trg_lens'])
    print(batch['abs_bins'])
    exit()
    """

    start_time = time.time()

    if opt.multi_style:
        decoder_dist, h_t, attention_dist, encoder_final_state, coverage, delimiter_decoder_states, delimiter_decoder_states_lens = model(
            src, src_lens, trg, src_oov, max_num_oov, src_mask, style_label)
    elif 2 in opt.control_modes:
        decoder_dist, h_t, attention_dist, encoder_final_state, coverage, delimiter_decoder_states, delimiter_decoder_states_lens = model(
            src, src_lens, trg, src_oov, max_num_oov, src_mask, batch['exact_lens'])
    elif 8 in opt.control_modes:
        query_tensor = batch['query_tensor'].to(opt.device)
        query_mask = batch['query_mask'].to(opt.device)
        query_lens = batch['query_lens']
        decoder_dist, h_t, attention_dist, encoder_final_state, coverage, delimiter_decoder_states, delimiter_decoder_states_lens = model(
            src, src_lens, trg, query_tensor, query_lens, src_oov, max_num_oov, src_mask, query_mask, position_ids)
        # forward(src, src_lens, trg, query, query_lens, src_oov, max_num_oov, src_mask, query_mask, position_ids=None)
    else:
        decoder_dist, h_t, attention_dist, encoder_final_state, coverage, delimiter_decoder_states, delimiter_decoder_states_lens = model(
            src, src_lens, trg, src_oov, max_num_oov, src_mask, position_ids)
    forward_time = time_since(start_time)

    start_time = time.time()
    if opt.copy_attention:  # Compute the loss using target with oov words
        loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                         opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, opt.coverage_loss, delimiter_decoder_states, opt.orthogonal_loss, opt.lambda_orthogonal, delimiter_decoder_states_lens)
    else:  # Compute the loss using target without oov words
        loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage, opt.coverage_loss, delimiter_decoder_states, opt.orthogonal_loss, opt.lambda_orthogonal, delimiter_decoder_states_lens)

    loss_compute_time = time_since(start_time)

    total_trg_tokens = sum(trg_lens)

    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("src")
        print(src)
        print(src_oov)
        print(src_str_list)
        print(src_lens)
        print(src_mask)
        print("trg")
        print(trg)
        print(trg_oov)
        print(trg_sent_2d_list)
        print(trg_lens)
        print(trg_mask)
        print("oov list")
        print(oov_lists)
        print("Decoder")
        print(decoder_dist)
        print(h_t)
        print(attention_dist)
        raise ValueError("Loss is NaN")

    if opt.loss_normalization == "tokens": # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches': # use batch_size to normalize the loss
        normalization = batch_size
    else:
        raise ValueError('The type of loss normalization is invalid.')

    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    # back propagation on the normalized loss
    loss.div(normalization).backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        # grad_norm_after_clipping = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
        # logging.info('clip grad (%f -> %f)' % (grad_norm_before_clipping, grad_norm_after_clipping))

    optimizer.step()

    # construct a statistic object for the loss
    stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)

    return stat, decoder_dist.detach()
