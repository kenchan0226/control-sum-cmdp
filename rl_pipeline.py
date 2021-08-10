import torch.nn as nn
from utils.masked_loss import masked_cross_entropy
from utils.statistics import RewardStatistics, LagrangianStatistics
from utils.time_log import time_since
import time
from sequence_generator import SequenceGenerator
from utils.report import export_and_plot_train_and_valid_reward, export_lagrangian_stats
from model.seq2seq_exact_length_input import Seq2SeqModelExactLenInput
import sys
import logging
from validation import evaluate_reward
from utils.reward import *
import math
from utils import io
import os
from utils.io import tokenize
import nltk
from cytoolz import concat
from utils.cost import compute_batch_cost
from utils.io import remove_old_ckpts, remove_old_epoch_states
from utils.cost import *
import copy
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

EPS = 1e-8


def build_reward_object(reward_type, device):
    if reward_type == 0:
        reward_obj = MixedRougeReward(device)
    elif reward_type == 1:
        #reward_obj = RougeLReward(device)
        reward_obj = SummRougeLReward(device)
    elif reward_type == 2:
        reward_obj = BertScoreReward(device)
    elif reward_type == 3:
        reward_obj = Rouge2Reward(device)
    elif reward_type == 4:
        reward_obj = MixedRougeRewardLenPenalty(device)
    elif reward_type == 5:
        reward_obj = BertRougeReward(device)
    elif reward_type == 6:
        reward_obj = GOLCReward(device)
    elif reward_type == 7:
        reward_obj = BertRouge2Reward(device)
    elif reward_type == 8:
        reward_obj = BertRouge2RewardParallel(device)
    elif reward_type == 9:
        reward_obj = BertScoreRewardRescaled(device)
    elif reward_type == 10:
        reward_obj = Rouge2LReward(device)
    elif reward_type == 11:
        reward_obj = BertScoreRewardRescaledLengthRepeatPenalty(device)
    elif reward_type == 12:
        reward_obj = BertScoreRewardRescaledLengthRepeatPenaltyWeighted(device)
    elif reward_type == 13:
        reward_obj = BertScoreRewardRescaledLengthRepeatPenaltyWeightedNew(device)
    elif reward_type == 14:
        reward_obj = BertScoreRewardRescaledLengthRepeatPenaltyWeightedNew2(device)
    elif reward_type == 15:
        reward_obj = WeightedROUGELReward(device)
    elif reward_type == 16:
        reward_obj = BertScoreRewardRescaledExtractiveRepeatButPenaltyWeighted(device)
    elif reward_type == 17:
        reward_obj = BertScoreRewardRescaledQAF1EntityRepeatPenaltyWeighted(device)
    elif reward_type == 18:
        reward_obj = AbsWeightedBaseline(device)
    else:
        raise ValueError
    return reward_obj

def build_cost_objects(cost_types, device, batch_size, thresholds, model=None, pretrained_model_args=None):
    cost_objs = []
    for cost_type, threshold in zip(cost_types, thresholds):
        if cost_type == 0:
            cost_obj = ThreeGramRepeatLoss(device)
        elif cost_type == 1:
            cost_obj = MinLenCost(device)
        elif cost_type == 2:
            cost_obj = StyleDiscriminatorCost(device)
        elif cost_type == 3:
            cost_obj = BadEndingCost(device)
        elif cost_type == 4:
            cost_obj = CorruptedDiscriminatorCost(device)
        elif cost_type == 5:
            cost_obj = HighReadabilityCosts(device)
        elif cost_type == 6:
            cost_obj = LowReadabilityCosts(device)
        elif cost_type == 7:
            cost_obj = LengthBinConsistent(device)
        elif cost_type == 8:
            cost_obj = WordBasedDifficulty(device)
        elif cost_type == 9:
            cost_obj = WordBasedDifficulty(device, is_negative=True)
        elif cost_type == 10:
            cost_obj = NovelNgramFraction(device, n=3)
        elif cost_type == 11:
            cost_obj = NovelNgramFraction(device, n=3, is_negative=True)
        elif cost_type == 12:
            cost_obj = ExactLengthCost(device)
        elif cost_type == 13:
            cost_obj = LengthBinDistance(device, total_len_bins=10)
        elif cost_type == 14:
            cost_obj = ExactLengthCostDistance(device)
        elif cost_type == 15:
            cost_obj = ThreeGramRepeatFraction(device)
        elif cost_type == 16:
            cost_obj = AbsBinDistance(device, n=3)
        elif cost_type == 17:
            cost_obj = AbsBinDistance(device, n=2)
        elif cost_type == 18:
            cost_obj = NovelNgramFraction(device, n=2)
        elif cost_type == 19:
            cost_obj = NovelNgramFraction(device, n=2, is_negative=True)
        elif cost_type == 20:
            cost_obj = ExactLengthCostDistanceUnnormalized(device)
        elif cost_type == 21:
            cost_obj = LengthBinDistanceUnnormalized(device)
        elif cost_type == 22:
            cost_obj = ExtFragDensityBinDistance(device)
        elif cost_type == 23:
            cost_obj = SentenceFusionBinDistance(device)
        elif cost_type == 24:
            cost_obj = SentenceFusionBinDistanceMultiProcess(device, batch_size)
        elif cost_type == 25:
            reference_model = copy.deepcopy(model)
            TLDR_ids_list = pretrained_model_args['TLDR_ids_list']
            pad_idx = pretrained_model_args['pad_idx']
            eos_idx = pretrained_model_args['eos_idx']
            reference_policy = ReferencePolicy(device, reference_model, TLDR_ids_list, pad_idx, eos_idx)
            cost_obj = KLDivergenceCost(device, reference_policy)
        elif cost_type == 26:
            cost_obj = NegativeNamedEntityF1(device)
        elif cost_type == 27:
            cost_obj = NegativeNamedEntityClozeConfidence(device, threshold)
        elif cost_type == 28:
            cost_obj = NegativeNamedEntityQAF1(device, threshold)
        elif cost_type == 29:
            cost_obj = EntityRepeatCost(device)
        elif cost_type == 30:
            cost_obj = IncorrectBut(device)
        else:
            raise ValueError("No matched cost function type.")
        cost_objs.append(cost_obj)
    return cost_objs


def train_model(model, optimizer_rl, train_data_loader, valid_data_loader, opt, lagrangian_params=None, epoch_state_dict=None):
    exp = opt.exp.split('.')[0]
    print("ml_loss coefficient: {}".format(opt.ml_loss_coefficient))

    # make the code compatible when tensorboardX is not available
    """
    try:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter()
        print("tb_writer")
        print(tb_writer)
        if tb_writer == None:
            raise ValueError
    except ModuleNotFoundError:
        print("No tensorboard!")
        tb_writer = None
    """
    tb_writer = SummaryWriter()
    logging.info('======================  Start Training  =========================')

    early_stop_flag = False
    if opt.train_from:
        report_train_reward_statistics = epoch_state_dict['report_train_reward_statistics']
        report_train_reward = epoch_state_dict['report_train_reward']
        report_valid_reward = epoch_state_dict['report_valid_reward']
        best_valid_reward = epoch_state_dict['best_valid_reward']
        previous_valid_reward = epoch_state_dict['previous_valid_reward']
        num_stop_increasing = epoch_state_dict['num_stop_increasing']
        total_batch = epoch_state_dict['total_batch']
    else:
        report_train_reward_statistics = RewardStatistics()
        #total_train_reward_statistics = RewardStatistics()
        report_train_reward = []
        report_valid_reward = []
        best_valid_reward = float('-inf')
        previous_valid_reward = float('-inf')
        num_stop_increasing = 0
        total_batch = -1

    reward_obj = build_reward_object(opt.reward_type, opt.device)

    cost_objs = []
    if opt.constrained_mdp:
        if opt.train_from:
            report_train_lagrangian_statistics = epoch_state_dict['report_train_lagrangian_statistics']
            report_lagrangian_loss = epoch_state_dict['report_lagrangian_loss']
            report_lagrangian_multipliers = epoch_state_dict['report_lagrangian_multipliers']
            report_violate_amounts = epoch_state_dict['report_violate_amounts']
            report_lagrangian_grad_norms = epoch_state_dict['report_lagrangian_grad_norms']
        else:
            report_train_lagrangian_statistics = LagrangianStatistics()
            report_lagrangian_loss = []
            report_lagrangian_multipliers = []
            report_violate_amounts = []
            report_lagrangian_grad_norms = []
        lagrangian_model, optimizer_lagrangian = lagrangian_params
        cost_objs = build_cost_objects(opt.cost_types, opt.device, opt.batch_size, opt.cost_thresholds)

    generator = SequenceGenerator(model,
                                  bos_idx=io.BOS,
                                  eos_idx=io.EOS,
                                  pad_idx=io.PAD,
                                  beam_size=1,
                                  max_sequence_length=opt.pred_max_len,
                                  cuda=opt.gpuid > -1,
                                  n_best=1,
                                  len_idx=opt.word2idx[io.EXACT_LEN_WORD] if 2 in opt.control_modes else -1
                                  )

    model.train()

    for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
        if early_stop_flag:
            break

        epoch_start_time = time.time()

        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, reward_obj, opt, total_batch, tb_writer, lagrangian_params, cost_objs)
            if opt.constrained_mdp:
                batch_reward_stat, batch_lagrangian_stat = stat
            else:
                batch_reward_stat = stat

            report_train_reward_statistics.update(batch_reward_stat)
            #total_train_reward_statistics.update(batch_reward_stat)
            if opt.constrained_mdp:
                report_train_lagrangian_statistics.update(batch_lagrangian_stat)

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
                    # log training reward and pg loss
                    current_train_reward = report_train_reward_statistics.reward()
                    current_train_pg_loss = report_train_reward_statistics.loss()
                    report_train_reward.append(current_train_reward)
                    # Run validation and log valid reward
                    valid_reward_stat = evaluate_reward(valid_data_loader, generator, reward_obj, opt)
                    model.train()
                    current_valid_reward = valid_reward_stat.reward()
                    report_valid_reward.append(current_valid_reward)
                    # print out train and valid reward
                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'avg training reward: %.4f; avg training loss: %.4f; avg validation reward: %.4f; best validation reward: %.4f' % (
                            current_train_reward, current_train_pg_loss, current_valid_reward, best_valid_reward))
                    # log lagrangian training loss and last lagrangian value
                    if opt.constrained_mdp:
                        current_lagrangian_loss = report_train_lagrangian_statistics.loss()
                        current_lagrangian_grad_norm = report_train_lagrangian_statistics.grad_norm()
                        current_violate_amount = report_train_lagrangian_statistics.violate_amt()
                        report_lagrangian_loss.append(current_lagrangian_loss)
                        report_violate_amounts.append(current_violate_amount)
                        report_lagrangian_grad_norms.append(current_lagrangian_grad_norm)
                        lagrangian_multipliers_array = lagrangian_model.get_lagrangian_multiplier_array()
                        report_lagrangian_multipliers.append(lagrangian_multipliers_array)
                        logging.info("Lagrangian_loss: %.5f; grad_norm: %.5f" % (current_lagrangian_loss, current_lagrangian_grad_norm))
                        logging.info("Value of lagrangian_multipliers: {}".format(lagrangian_multipliers_array))

                    if epoch >= opt.start_decay_and_early_stop_at:
                        if current_valid_reward > previous_valid_reward: # update the best valid reward and save the model parameters
                            logging.info("Valid reward increases")
                            sys.stdout.flush()
                            if current_valid_reward > best_valid_reward:
                                best_valid_reward = current_valid_reward
                            num_stop_increasing = 0

                            check_pt_model_path = os.path.join(opt.model_path, 'ckpt', '%s-epoch-%d-total_batch-%d-valid_reward-%.5f' % (
                                opt.exp, epoch, total_batch, current_valid_reward))
                            torch.save(  # save model parameters
                                model.state_dict(),
                                open(check_pt_model_path, 'wb')
                            )
                            logging.info('Saving checkpoint to %s' % check_pt_model_path)

                            # Only keep the highest three checkpoints
                            remove_old_ckpts(opt.model_path, reverse=True)
                        else:
                            print("Valid reward does not increase")
                            sys.stdout.flush()
                            num_stop_increasing += 1
                            # decay the learning rate by the factor specified by opt.learning_rate_decay
                            decay_learning_rate(optimizer_rl, opt.learning_rate_decay, opt.min_lr)

                        previous_valid_reward = current_valid_reward

                        # decay the learning rate for lagrangian multiplier
                        if opt.constrained_mdp and opt.decay_multiplier_learning_rate:
                            logging.info("Decay learning rate of lagrangian multiplier....")
                            decay_learning_rate(optimizer_lagrangian, 0.5, 1e-8)

                        if not opt.disable_early_stop:
                            if num_stop_increasing >= opt.early_stop_tolerance:
                                logging.info('Have not increased for %d check points, early stop training' % num_stop_increasing)
                                early_stop_flag = True
                                break

                    report_train_reward_statistics.clear()
                    if opt.constrained_mdp:
                        report_train_lagrangian_statistics.clear()
        # save the model and optimizer state for further training
        if epoch % 2 == 0:
            # save epoch state
            epoch_state = {
                'epoch': epoch,
                'total_batch': total_batch,
                'model': model.state_dict(),
                'optimizer_rl': optimizer_rl.state_dict(),
                'lagrangian_model': lagrangian_model.state_dict() if opt.constrained_mdp else None,
                'optimizer_lagrangian': optimizer_lagrangian.state_dict() if opt.constrained_mdp else None,
                'best_valid_reward': best_valid_reward,
                'previous_valid_reward': previous_valid_reward,
                'num_stop_increasing': num_stop_increasing,
                'report_train_reward_statistics': report_train_reward_statistics,
                'report_train_reward': report_train_reward,
                'report_valid_reward': report_valid_reward,
                'report_train_lagrangian_statistics': report_train_lagrangian_statistics if opt.constrained_mdp else None,
                'report_lagrangian_loss': report_lagrangian_loss if opt.constrained_mdp else None,
                'report_lagrangian_multipliers': report_lagrangian_multipliers if opt.constrained_mdp else None,
                'report_violate_amounts': report_violate_amounts if opt.constrained_mdp else None,
                'report_lagrangian_grad_norms': report_lagrangian_grad_norms if opt.constrained_mdp else None,
            }
            epoch_state_path = os.path.join(opt.model_path, 'epoch_states', '{}-epoch.pt'.format(epoch))
            torch.save(  # save epoch states
                epoch_state,
                open(epoch_state_path, 'wb')
            )
            logging.info("saved epoch state dict.")
            # remove old epoch states
            remove_old_epoch_states(os.path.join(opt.model_path, 'epoch_states'))

        # print("epoch time: {}".format(time_since(epoch_start_time)))

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_and_plot_train_and_valid_reward(report_train_reward, report_valid_reward, opt.checkpoint_interval, train_valid_curve_path)
    if opt.constrained_mdp:
        export_lagrangian_stats(report_lagrangian_loss, report_lagrangian_multipliers, report_lagrangian_grad_norms, report_violate_amounts, opt.checkpoint_interval, opt.exp_path)

    # log best reward
    logging.info("final_best_valid_reward: %.3f" % best_valid_reward)

    # print epoch states
    logging.info("epoch states folder: {}".format(os.path.join(opt.model_path, 'epoch_states')))

def train_one_batch(batch, generator, optimizer, reward_obj, opt, global_step, tb_writer, lagrangian_params=None, cost_objs=[]):
    #src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_sent_2d_list, trg, trg_oov, trg_lens, trg_mask, _ = batch
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
    trg_sent_2d_list = batch['trg_sent_2d_list']
    src_sent_2d_list = batch['src_sent_2d_list']
    trg = batch['trg_tensor']
    trg_oov = batch['trg_oov_tensor']
    trg_lens = batch['trg_lens']
    trg_mask = batch['trg_mask']

    control_variables = {}
    if 1 in opt.control_modes:
        control_variables['len_bins'] = batch['len_bins']
    if 2 in opt.control_modes:
        control_variables['exact_lens'] = batch['exact_lens']
    if 3 in opt.control_modes or 4 in opt.control_modes or 5 in opt.control_modes or 6 in opt.control_modes:
        control_variables['abs_bins'] = batch['abs_bins']
        if 6 in opt.control_modes:
            # tokenize the each src sentence list in the batch and put it in the control variable.
            src_sent_2d_list_tokenized = []  # each item is a list of src sentences (tokenized) for an input sample
            for src_sent_list in src_sent_2d_list:
                src_sent_list = [src_sent.split(' ') for src_sent in src_sent_list]
                src_sent_2d_list_tokenized.append(src_sent_list)
            control_variables['src_word_2d_list_sent_tokenized'] = src_sent_2d_list_tokenized
        #control_variables['src_word_2d_list'] = batch['src_list_tokenized']
    #if 10 in opt.cost_types or 11 in opt.cost_types or 18 in opt.cost_types or 19 in opt.cost_types:
    if 7 in opt.control_modes:
        control_variables['reference_entities_list'] = batch['reference_entities_list']
        position_ids = batch['position_ids']
        position_ids = position_ids.to(opt.device)
        control_variables['masked_questions_ids_2dlist'] = batch['masked_questions_ids_2dlist']
        control_variables['answer_2dlist'] = batch['answer_2dlist']
        #control_variables['answer_id_2dlist'] = batch['answer_id_2dlist']
        #control_variables['multiple_choices_ids_2dlist'] = batch['multiple_choices_ids_2dlist']
    else:
        position_ids = None
    control_variables['src_word_2d_list'] = batch['src_list_tokenized']

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)

    optimizer.zero_grad()

    # debug
    """
    for src_str in src_str_list:
        print(src_str[:10])
    print(src.detach().cpu().numpy()[:, :10])
    print(batch['trg_lens'])
    print(batch['len_bins'])
    print(batch['abs_bins'])
    exit()
    """

    batch_size = src.size(0)
    reward_type = opt.reward_type
    sent_level_reward = opt.sent_level_reward
    baseline = opt.baseline
    regularization_type = opt.regularization_type
    regularization_factor = opt.regularization_factor

    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False

    trg_sent_2d_list_tokenized = []  # each item is a list of target sentences (tokenized) for an input sample
    trg_str_list = []  # each item is the target output sequence (tokenized) for an input sample
    for trg_sent_list in trg_sent_2d_list:
        trg_sent_list = [trg_sent.strip().split(' ') for trg_sent in trg_sent_list]
        trg_sent_2d_list_tokenized.append(trg_sent_list)
        trg_str_list.append(list(concat(trg_sent_list)))

    trg_sent_2d_list = trg_sent_2d_list_tokenized  # each item is a list of target sentences (tokenized) for an input sample

    # if use self critical as baseline, greedily decode a sequence from the model
    if baseline == 'self':
        # sample greedy prediction
        generator.model.eval()
        with torch.no_grad():
            if isinstance(generator.model, Seq2SeqModelExactLenInput):
                greedy_sample_list, _, _, greedy_eos_idx_mask, _, _ = generator.sample_with_exact_len_input(src, src_lens, src_oov, src_mask, oov_lists, batch['exact_lens'], greedy=True,
                                                                                   entropy_regularize=False)
            else:
                greedy_sample_list, _, _, greedy_eos_idx_mask, _, _ = generator.sample(src, src_lens, src_oov, src_mask,
                                                                                       oov_lists, greedy=True,
                                                                                       entropy_regularize=False,
                                                                                       position_ids=position_ids)

            greedy_str_list = sample_list_to_str_list(greedy_sample_list, oov_lists, opt.idx2word, opt.vocab_size,
                                                      io.EOS,
                                                      io.UNK, opt.replace_unk,
                                                      src_str_list)
            greedy_sent_2d_list = []
            for greedy_str in greedy_str_list:
                greedy_sent_list = nltk.tokenize.sent_tokenize(' '.join(greedy_str))
                greedy_sent_list = [greedy_sent.strip().split(' ') for greedy_sent in greedy_sent_list]
                greedy_sent_2d_list.append(greedy_sent_list)

            # compute reward of greedily decoded sequence, tensor with size [batch_size]
            baseline = compute_batch_reward(greedy_str_list, greedy_sent_2d_list, trg_str_list,
                                            trg_sent_2d_list, batch_size, reward_obj,
                                            regularization_factor=0.0, regularization_type=0, entropy=None, control_variables=control_variables)
        generator.model.train()

    if opt.ml_loss_coefficient > 0:
        generator.model.train()
        total_trg_tokens = sum(trg_lens)
        trg = trg.to(opt.device)
        trg_mask = trg_mask.to(opt.device)
        trg_oov = trg_oov.to(opt.device)
        ml_loss = compute_ml_loss(generator.model, src, src_lens, src_mask, src_oov, oov_lists, trg, trg_oov, trg_lens, trg_mask, generator.model.copy_attn, opt.control_modes, control_variables)
        ml_loss_normalized = ml_loss.div(total_trg_tokens)
        #ml_loss_normalized = ml_loss.div(batch_size)

    # sample a sequence from the model
    # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, prediction is a list of 0 dim tensors
    # log_selected_token_dist: size: [batch, output_seq_len]

    # sample sequences for multiple times
    sample_batch_size = batch_size * opt.n_sample
    src = src.repeat(opt.n_sample, 1)
    src_lens = src_lens * opt.n_sample
    src_mask = src_mask.repeat(opt.n_sample, 1)
    src_oov = src_oov.repeat(opt.n_sample, 1)
    oov_lists = oov_lists * opt.n_sample
    src_str_list = src_str_list * opt.n_sample
    trg_sent_2d_list = trg_sent_2d_list * opt.n_sample
    trg_str_list = trg_str_list * opt.n_sample
    if opt.baseline != 'none':  # repeat the greedy rewards
        #baseline = np.tile(baseline, opt.n_sample)
        baseline = baseline.repeat(opt.n_sample)  # [sample_batch_size]

    start_time = time.time()
    if isinstance(generator.model, Seq2SeqModelExactLenInput):
        repeated_exact_lens = batch['exact_lens'] * opt.n_sample
        sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch = \
            generator.sample_with_exact_len_input(src, src_lens, src_oov, src_mask, oov_lists, repeated_exact_lens, greedy=False, entropy_regularize=entropy_regularize)
    else:
        sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch = generator.sample(
            src, src_lens, src_oov, src_mask, oov_lists, greedy=False, entropy_regularize=entropy_regularize, position_ids=position_ids)
    pred_str_list = sample_list_to_str_list(sample_list, oov_lists, opt.idx2word, opt.vocab_size, io.EOS,
        io.UNK, opt.replace_unk, src_str_list)  # a list of word list, len(pred_word_2dlist)=sample_batch_size
    sample_time = time_since(start_time)
    max_pred_seq_len = log_selected_token_dist.size(1)

    pred_sent_2d_list = []  # each item is a list of predicted sentences (tokenized) for an input sample, used to compute summary level Rouge-l
    for pred_str in pred_str_list:
        pred_sent_list = nltk.tokenize.sent_tokenize(' '.join(pred_str))
        pred_sent_list = [pred_sent.strip().split(' ') for pred_sent in pred_sent_list]
        pred_sent_2d_list.append(pred_sent_list)

    if entropy_regularize:
        entropy_array = entropy.data.cpu().numpy()
    else:
        entropy_array = None

    # compute the reward
    with torch.no_grad():
        if sent_level_reward:
            raise ValueError("Not implemented.")
        else:  # neither using reward shaping
            # only receive reward at the end of whole sequence, tensor: [sample_batch_size]
            cumulative_reward = compute_batch_reward(pred_str_list, pred_sent_2d_list, trg_str_list,
                                            trg_sent_2d_list, sample_batch_size, reward_obj,
                                            regularization_factor=regularization_factor,
                                                     regularization_type=regularization_type, entropy=entropy_array,
                                                     control_variables=control_variables)


            # store the sum of cumulative reward (before baseline) for the experiment log
            cumulative_reward_sum = cumulative_reward.detach().sum(0).item()

            if opt.constrained_mdp:
                lagrangian_model, optimizer_lagrangian = lagrangian_params
                cumulative_cost = compute_batch_cost(pred_str_list, pred_sent_2d_list, trg_str_list, trg_sent_2d_list, sample_batch_size, cost_objs, control_variables)  # [sample_batch_size, num_cost_types]
                cumulative_cost_mean = cumulative_cost.mean(0)  # [num_cost_types]
                #cumulative_cost = torch.from_numpy(cumulative_cost_array).type(torch.FloatTensor).to(src.device)

                # cumulative_cost: [sample_batch_size, len(cost_types)]
                # subtract the regularization term: \lambda \dot C_t
                constraint_regularization = lagrangian_model.compute_regularization(cumulative_cost)  # [sample_batch_size]
                cumulative_reward -= constraint_regularization

            # Subtract the cumulative reward by a baseline if needed
            if opt.baseline != 'none':
                cumulative_reward = cumulative_reward - baseline  # [sample_batch_size]
            # q value estimation for each time step equals to the (baselined) cumulative reward
            q_value_estimate = cumulative_reward.unsqueeze(1).repeat(1, max_pred_seq_len)  # [sample_batch_size, max_pred_seq_len]
            #q_value_estimate_array = np.tile(cumulative_reward.reshape([-1, 1]), [1, max_pred_seq_len])  # [batch, max_pred_seq_len]

    #shapped_baselined_reward = torch.gather(shapped_baselined_phrase_reward, dim=1, index=pred_phrase_idx_mask)

    # use the return as the estimation of q_value at each step

    #q_value_estimate = torch.from_numpy(q_value_estimate_array).type(torch.FloatTensor).to(src.device)
    q_value_estimate.requires_grad_(True)
    q_estimate_compute_time = time_since(start_time)

    # compute the policy gradient objective
    pg_loss = compute_pg_loss(log_selected_token_dist, output_mask, q_value_estimate)

    # back propagation to compute the gradient
    if opt.loss_normalization == "samples": # use number of target tokens to normalize the loss
        normalization = opt.n_sample
    elif opt.loss_normalization == 'batches': # use batch_size to normalize the loss
        normalization = sample_batch_size
    else:
        normalization = 1

    pg_loss_normalized = pg_loss.div(normalization)

    if opt.ml_loss_coefficient > 0:
        if opt.pg_loss_coefficient > 0:
            total_loss = opt.pg_loss_coefficient * pg_loss_normalized + opt.ml_loss_coefficient * ml_loss_normalized
        else:
            total_loss = (1 - opt.ml_loss_coefficient) * pg_loss_normalized + opt.ml_loss_coefficient * ml_loss_normalized
    else:
        ml_loss_normalized = torch.Tensor([0.0])
        total_loss = pg_loss_normalized

    start_time = time.time()
    total_loss.backward()
    #pg_loss.div(normalization).backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(generator.model.parameters(), opt.max_grad_norm)

    # take a step of gradient descent
    optimizer.step()

    #print(ml_loss_normalized.item())
    #print(pg_loss_normalized.item())

    # log each loss to tensorboard
    if tb_writer is not None:
        tb_writer.add_scalar('ml_loss', ml_loss_normalized.item(), global_step)
        tb_writer.add_scalar('pg_loss', pg_loss_normalized.item(), global_step)
        if opt.constrained_mdp:
            lambda_tensor = lagrangian_model.get_lagrangian_multiplier()
            for cost_i, cost_type in enumerate(opt.cost_types):
                tb_writer.add_scalar('cost_{}'.format(cost_type), cumulative_cost_mean[cost_i].detach().item(), global_step)
                tb_writer.add_scalar('lambda_{}'.format(cost_type), lambda_tensor[cost_i].item(), global_step)

    stat = RewardStatistics(cumulative_reward_sum, pg_loss.item(), sample_batch_size, sample_time, q_estimate_compute_time, backward_time)
    # (final_reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0)
    # reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0

    if opt.constrained_mdp:
        lagrangian_loss, lagrangian_grad_norm, violate_amount = train_lagrangian_multiplier(lagrangian_model, cumulative_cost, optimizer_lagrangian, normalization, opt.max_grad_norm)
        lagrangian_stat = LagrangianStatistics(lagrangian_loss=lagrangian_loss, n_batch=sample_batch_size, lagrangian_grad_norm=lagrangian_grad_norm, violate_amount=violate_amount)
        stat = (stat, lagrangian_stat)

    return stat, log_selected_token_dist.detach()


def compute_ml_loss(model, src, src_lens, src_mask, src_oov, oov_lists, trg, trg_oov, trg_lens, trg_mask,
                    copy_attention, control_modes=[], control_variables=None):

    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
    if isinstance(model, Seq2SeqModelExactLenInput):
        decoder_dist, h_t, attention_dist, encoder_final_state, coverage, delimiter_decoder_states, delimiter_decoder_states_lens = model(
            src, src_lens, trg, src_oov, max_num_oov, src_mask, control_variables['exact_lens'])
    else:
        decoder_dist, h_t, attention_dist, encoder_final_state, coverage, delimiter_decoder_states, delimiter_decoder_states_lens = model(
            src, src_lens, trg, src_oov, max_num_oov, src_mask)

    if copy_attention:  # Compute the loss using target with oov words
        loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens)
    else:  # Compute the loss using target without oov words
        loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens)

    return loss

def train_lagrangian_multiplier(lagrangian_model, cumulative_cost, optimizer, normalization, max_grad_norm):
    """
    :param lagrangian_multiplier: [batch, len(cost_types)]
    :param cumulative_cost: [batch, len(cost_types)]
    :param cost_threshold: [len(cost_types)]
    :param optimizer:
    :param normalization
    :return:
    """
    optimizer.zero_grad()
    lagrangian_loss, violate_amount = lagrangian_model(cumulative_cost)
    lagrangian_loss.div(normalization).backward()
    grad_norm = lagrangian_model.lagrangian_multiplier.grad.detach().sum().item()
    #grad_norm = lagrangian_model.lagrangian_multiplier.grad.detach().norm(2).item()
    #grad_norm_before_clipping = nn.utils.clip_grad_norm_(lagrangian_model.parameters(), max_grad_norm)
    optimizer.step()
    lagrangian_model.clamp_lagrangian_multiplier()
    return lagrangian_loss.item(), grad_norm, violate_amount.item()

def decay_learning_rate(optimizer, decay_factor, min_lr):
    # decay the learning rate by the factor specified by opt.learning_rate_decay
    if decay_factor < 1:
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * decay_factor
            if new_lr < min_lr:
                new_lr = min_lr
            if old_lr - new_lr > EPS:
                param_group['lr'] = new_lr
        logging.info('Learning rate drops to {}'.format(new_lr))

