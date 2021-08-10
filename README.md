# Controllable Summarization with Constrained Markov Decision Process

This repository contains the source code for our TACL paper "[Controllable Summarization with Constrained Markov Decision Process](https://arxiv.org/abs/2108.03405)". 

Some of our code are adapted from [Huggingface Transformers](https://github.com/huggingface/transformers), [Fast Abstractive Summarization-RL](https://github.com/ChenRocks/fast_abs_rl), and [summa-qa](https://github.com/recitalAI/summa-qa). 

If you use our code, please cite our paper:
```
@article{DBLP:journals/tacl/cmdp-control-sum-21,
  author    = {Hou Pong Chan and Lu Wang and Irwin King},
  title     = {Controllable Summarization with Constrained Markov Decision Process},
  journal   = {Transactions of the Association for Computational Linguistics},
  year      = {2021}
}
```

## Dependencies
- python==3.7
- pytorch==1.4.0
- transformers==2.5.0
- bert-score==0.3.1
- moverscore==1.0.3
- gensim
- cytoolz
- pyrouge
- ntlk
- tqdm
- rreplace
- sklearn
- tensorflow==2.1.0 (for tensorboard)
- tensorboardx
- spacy==2.1.9
- neuralcoref

## Datasets
You can download the preprocessed datasets as follows
- [CNN/DM](https://www.dropbox.com/s/nguzf75q0e85qjy/cased-cnn-dailymail.tar.gz?dl=0), for length and abstractiveness control. 
- [Newsroom-b](https://www.dropbox.com/s/hv2vyab1izqnv7b/newsroom_guardian_ws_nyt.tar.gz?dl=0)
- [CNN/DM](https://www.dropbox.com/s/c1xhc8y25bmytof/cased-cnn-dailymail_coref_3.tar.gz?dl=0), for entity control, the data for training the QA model of QAF1 score is located in the `cloze_entity_squad_with_idx_and_unanswerable_and_paraphrase_and_repeat` folder. 
- For the DUC-2002 dataset, please sign the agreements and request the DUC-2002 dataset follows the instructions [here](https://duc.nist.gov/data.html). After you obtain their approval, please send an email to me (hpchan@um.edu.mo) to request our preprocessed version of DUC-2002.

## Length Control
### Training
- ML training for D.GPT2 model on CNN/DM. 
```
python3 -m torch.distributed.launch \
--master_port=1234 \
--nproc_per_node 4 gpt2_summarization_finetuning.py \
--data_dir ../../datasets/cased-cnn-dailymail \
--output_dir saved_models/train_ml_distilgpt2_cnn_512_control_mode_1 \
--model_type gpt2 \
--model_name_or_path distilgpt2 \
--tokenizer_name distilgpt2 \
--cache_dir /data/model_cache \
--do_train \
--evaluate_during_training \
--per_gpu_train_batch_size 6 \
--per_gpu_eval_batch_size 16 \
--learning_rate 5e-5 \
--logging_steps 5000 \
--seed 9527 \
--num_train_epochs 12 \
--max_epochs 12 \
--control_modes 1 \
--gradient_accumulation_steps 3 \
--input_trunc_length 512 \
--fp16
```
- CMDP training for D.GPT2 model on CNN/DM. 
```
python3 -m torch.distributed.launch \
--master_port=1234 \
--nproc_per_node 4 gpt2_summarization_rl_finetuning.py \
--data_dir ../../datasets/cased-cnn-dailymail \
--output_dir saved_models/train_rl_distilgpt2_cnn_bertscore_control_mode_1_512 \
--model_type gpt2 \
--model_name_or_path saved_models/train_ml_distilgpt2_cnn_512_control_mode_1/epoch_states/10-epoch \
--tokenizer_name saved_models/train_ml_distilgpt2_cnn_512_control_mode_1 \
--cache_dir /data/model_cache \
--do_train \
--evaluate_during_training \
--per_gpu_train_batch_size 12 \
--per_gpu_eval_batch_size 24 \
--learning_rate 1.77e-5 \
--save_total_limit 1 \
--max_output_length 90 \
--seed 9527 \
--num_train_epochs 20 \
--reward_type 9 \
--baseline self \
--constrained_mdp \
--cost_types 13 15 \
--cost_thresholds 0.0 0.0 \
--lagrangian_init_val 0.01 \
--logging_steps 1000 \
--control_modes 1 \
--gradient_accumulation_steps 3 \
--fp16 \
--num_freeze_layers 4
```
### Testing
- Download pyrouge, and save it to `path/to/pyrouge`. 
`git clone https://github.com/andersjo/pyrouge.git`
- Export ROUGE score environment variable
`export ROUGE=[path/to/pyrouge/tools/ROUGE-1.5.5]`
- Make evaluation reference for CNN/DM and DUC2002 datasets (only need to do it for once for each dataset)
`python make_eval_reference.py -data ../../datasets/cased-cnn-dailymail -split test`
`python make_eval_reference_duc.py -data ../../datasets/duc2002_preprocessed_long -split test`

#### Reference length bin control
- Decode from D.GPT2 model using reference length bins on CNN/DM. Specify the `model_name_or_path` to the best checkpoint (highest validation reward). 
```
python3 -u gpt2_summarization_prediction.py \
--model_type gpt2 \
--model_name_or_path saved_models/train_rl_distilgpt2_cnn_bertscore_control_mode_1_512/epoch_states/18-epoch \
--tokenizer_name saved_models/train_rl_distilgpt2_cnn_bertscore_control_mode_1_512 \
--pred_path pred/predict_rl_distilgpt2_cnn_512_control_mode_1 \
--data_dir ../../datasets/cased-cnn-dailymail \
--split test \
--temperature 1.0 \
--batch_size 8 \
--beam_size 5 \
--control_modes 1 \
--with_ground_truth_input \
--seed 9527 \
--input_trunc_length 512
```
- Compute ROUGE scores
```
python evaluate_prediction.py -rouge -decode_dir pred/predict_rl_distilgpt2_cnn_512_control_mode_1 -data ../../datasets/cased-cnn-dailymail
```
- Compute bin %
```
python output_len_stat.py -decode_dir pred/predict_rl_distilgpt2_cnn_512_control_mode_1 -data_dir ../../datasets/cased-cnn-dailymail -split test
```
#### Arbitrary length bin control
- Decode from D.GPT2 model using a particular length bin on DUC2002. Set `--desired_target_number` to the desired length bin - 1. E.g., if the desired length bin is 1, you should set `--desired_target_number 0`. 
```
python3 -u gpt2_summarization_prediction.py \
--model_type gpt2 \
--model_name_or_path saved_models/train_rl_distilgpt2_cnn_bertscore_control_mode_1_512/epoch_states/18-epoch \
--tokenizer_name saved_models/train_rl_distilgpt2_cnn_bertscore_control_mode_1_512 \
--pred_path pred/predict_rl_distilgpt2_duc_512_control_mode_1_bin_1 \
--data_dir ../../datasets/duc2002_preprocessed_long \
--split test \
--temperature 1.0 \
--batch_size 8 \
--beam_size 5 \
--control_modes 1 \
--desired_target_number 0 \
--multiple_reference \
--seed 9527 \
--input_trunc_length 512
```
- Compute ROUGE scores for bin 1, bin4, bin 7, and bin 10 respectively.
```
python evaluate_prediction.py -rouge -decode_dir pred/predict_rl_distilgpt2_duc_512_control_mode_1_epoch_18_bin_1 -data ../../datasets/duc2002_preprocessed_long -multi_ref -n_words 33
python evaluate_prediction.py -rouge -decode_dir pred/predict_rl_distilgpt2_duc_512_control_mode_1_epoch_18_bin_4 -data ../../datasets/duc2002_preprocessed_long -multi_ref -n_words 46
python evaluate_prediction.py -rouge -decode_dir pred/predict_rl_distilgpt2_duc_512_control_mode_1_bin_7 -data ../../datasets/duc2002_preprocessed_long -multi_ref -n_words 59
python evaluate_prediction.py -rouge -decode_dir pred/predict_rl_distilgpt2_duc_512_control_mode_1_bin_10 -data ../../datasets/duc2002_preprocessed_long -multi_ref -n_words 94
```
- Compute bin % for bin 1, bin4, bin 7, and bin 10 respectively.
```
python output_len_stat.py -decode_dir pred/predict_rl_distilgpt2_duc_512_control_mode_1_epoch_18_bin_1 -data_dir ../../datasets/duc2002_preprocessed_long -split test -target_len_bin 9 -multi_ref
python output_len_stat.py -decode_dir pred/predict_rl_distilgpt2_duc_512_control_mode_1_epoch_18_bin_4 -data_dir ../../datasets/duc2002_preprocessed_long -split test -target_len_bin 6 -multi_ref
python output_len_stat.py -decode_dir pred/predict_rl_distilgpt2_duc_512_control_mode_1_epoch_18_bin_7 -data_dir ../../datasets/duc2002_preprocessed_long -split test -target_len_bin 3 -multi_ref
python output_len_stat.py -decode_dir pred/predict_rl_distilgpt2_duc_512_control_mode_1_epoch_18_bin_10 -data_dir ../../datasets/duc2002_preprocessed_long -split test -target_len_bin 0 -multi_ref
```
## Entity control
### Preprocessing
- Download QA model [here](https://www.dropbox.com/s/zn5waomygipuvr4/qaf1_score_model.tar.gz?dl=0) for QA-F1 score. 
  Then put this tar.gz file in the folder `./saved_models` and decompress it. If you want to put the QA model in another path, 
  please change the `self.model_path` and `self.tokenizer_path` in line 17 and 18 of `utils/cloze_model.py`.
- (Optional) Alternatively, you can also train a QA model by yourself and pick the best checkpoint. 
```
python3 -m torch.distributed.launch \
--master_port=1234 \
--nproc_per_node 4 train_cloze_model.py \
--data_dir ../../datasets/cased-cnn-dailymail_coref_3/cloze_entity_squad_with_idx_and_unanswerable_and_paraphrase_and_repeat \
--model_type bert \
--model_name_or_path twmkn9/bert-base-uncased-squad2 \
--output_dir saved_models/entity_cloze_with_unanswerable_paraphrase_repeat_bert_base_cnn_epoch_16 \
--cache_dir /mnt/sharedfolder/hpchan \
--max_seq_length 500 \
--do_train \
--do_eval \
--evaluate_during_training \
--do_lower_case \
--per_gpu_train_batch_size 11 \
--per_gpu_eval_batch_size 24 \
--gradient_accumulation_steps 1 \
--num_train_epochs 16 \
--logging_steps 8000 \
--save_steps 8000 \
--eval_all_checkpoints \
--fp16
```
### Training
- ML training for D.GPT2 model. 
```
python3 -m torch.distributed.launch \
--master_port=1234 \
--nproc_per_node 4 gpt2_summarization_finetuning.py \
--data_dir ../../datasets/cased-cnn-dailymail_coref_3 \
--output_dir saved_models/train_ml_distilgpt2_cnn_control_mode_7_batch_10_4gpu_fp_16_epoch12_entity_first_512_no_datetime \
--model_type gpt2 \
--model_name_or_path distilgpt2 \
--tokenizer_name distilgpt2 \
--cache_dir /data/model_cache \
--do_train \
--evaluate_during_training \
--per_gpu_train_batch_size 6 \
--per_gpu_eval_batch_size 16 \
--learning_rate 5e-5 \
--logging_steps 5000 \
--seed 9527 \
--num_train_epochs 12 \
--max_epochs 12 \
--control_modes 7 \
--gradient_accumulation_steps 3 \
--fp16
```
- RL training for D.GPT2 model. 
```
python3 -m torch.distributed.launch \
--master_port=1234 \
--nproc_per_node 4 gpt2_summarization_rl_finetuning.py \
--data_dir ../../datasets/cased-cnn-dailymail_coref_3 \
--output_dir saved_models/train_distilgpt2_rl_cnn_512_control_mode_7_bertscore_reward_QAF1_1.0_entity_repeat_epoch20 \
--model_type gpt2 \
--model_name_or_path saved_models/train_ml_distilgpt2_cnn_control_mode_7_batch_10_4gpu_fp_16_epoch12_entity_first_512_no_datetime_from_10/epoch_states/12-epoch \
--tokenizer_name saved_models/train_ml_distilgpt2_cnn_control_mode_7_batch_10_4gpu_fp_16_epoch12_entity_first_512_no_datetime_from_10 \
--cache_dir /mnt/sharedfolder/hpchan \
--do_train \
--evaluate_during_training \
--per_gpu_train_batch_size 12 \
--per_gpu_eval_batch_size 24 \
--learning_rate 1.77e-5 \
--max_output_length 75 \
--seed 9527 \
--num_train_epochs 20 \
--reward_type 9 \
--baseline self \
--constrained_mdp \
--cost_types 28 29 15 \
--cost_thresholds 0.0 -0.9 0.0 \
--lagrangian_init_val 0.01 \
--logging_steps 1000 \
--control_modes 7 \
--gradient_accumulation_steps 3 \
--fp16 \
--num_freeze_layers 4
```

### Testing
- Download and export the path of ROUGE following the testing procedure of length control.

#### Reference entity control
- Make evaluation reference for dataset. 
`python make_eval_reference.py -data ../../datasets/cased-cnn-dailymail_coref_3 -split test`
- Decode from D.GPT2 model using reference entities as input. Specify the `model_name_or_path` to the best checkpoint (highest validation reward). 
```
python3 -u gpt2_summarization_prediction.py \
--model_type gpt2 \
--model_name_or_path saved_models/train_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_QAF1_0.9_epoch20_from_7/epoch_states/20-epoch \
--tokenizer_name saved_models/train_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_QAF1_0.9_epoch20_from_7 \
--pred_path pred/predict_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_cloze_squad_cost_with_negative_0.9 \
--data_dir ../../datasets/cased-cnn-dailymail_coref_3 \
--split test \
--temperature 1.0 \
--batch_size 8 \
--beam_size 5 \
--control_modes 7 \
--with_ground_truth_input \
--seed 9527 \
--input_trunc_length 512
```
- Compute ROUGE Scores
`python evaluate_prediction.py -rouge -decode_dir pred/predict_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_cloze_squad_cost_with_negative_0.9 -data ../../datasets/cased-cnn-dailymail_coref_3`
- Compute Appear %
`python evaluate_entity_appear.py --decode_dir pred/predict_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_cloze_squad_cost_with_negative_0.9 --data ../../datasets/cased-cnn-dailymail_coref_3`
- Compute QA-F1
`python evaluate_cloze.py --decode_dir pred/predict_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_cloze_squad_cost_with_negative_0.9 --data ../../datasets/cased-cnn-dailymail_coref_3`

#### Entities at different document sentences
- Decode from D.GPT2 model for entities at particular document sentences. The option `--split test_1to2_sent_ent` indicates that we are summarizing the entities in sentences 1 and 2. Alternatively, you can set `--split` to `test_3to4_sent_ent`, `test_5to6_sent_ent`, or `test_7to8_sent_ent`. 
```
python3 -u gpt2_summarization_prediction.py \
--model_type gpt2 \
--model_name_or_path saved_models/train_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_QAF1_0.9_entity_repeat_epoch20_from_7/epoch_states/9-epoch \
--tokenizer_name saved_models/train_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_QAF1_0.9_entity_repeat_epoch20_from_7 \
--pred_path pred/predict_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_QAF1_0.9_with_repeat_epoch9_sent_12 \
--data_dir ../../datasets/cased-cnn-dailymail_coref_3 \
--split test \
--temperature 1.0 \
--batch_size 8 \
--beam_size 5 \
--control_modes 7 \
--with_ground_truth_input \
--split test_1to2_sent_ent \
--seed 9527 \
--input_trunc_length 512
```
- Compute Appear %
`python evaluate_entity_appear.py --decode_dir pred/predict_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_QAF1_0.9_with_repeat_epoch9_sent_12 --data ../../datasets/cased-cnn-dailymail_coref_3`
- Compute QA'-F1
`python evaluate_cloze.py --decode_dir pred/predict_distilgpt2_rl_cnn_512_rouge2_control_mode_7_bertscore_reward_QAF1_0.9_with_repeat_epoch9_sent_12 --data ../../datasets/cased-cnn-dailymail_coref_3`
  
## Abstractivenes Control
### Training
- ML training for D.GPT2 model on Newsroom-b, change `--data_dir` to the path of `cased-cnn-dailymail` if you want to use CNN/DM. 
```
python3 -m torch.distributed.launch \
--master_port=1234 \
--nproc_per_node 4 gpt2_summarization_finetuning.py \
--data_dir ../../datasets/newsroom_guardian_ws_nyt \
--output_dir saved_models/train_ml_distilgpt2_cnn_mode_5_3_bins_batch4_6gpu_fp_16_epoch_10_512_tokens \
--model_type gpt2 \
--model_name_or_path distilgpt2 \
--tokenizer_name distilgpt2 \
--cache_dir /data/model_cache \
--do_train \
--evaluate_during_training \
--per_gpu_train_batch_size 6 \
--per_gpu_eval_batch_size 16 \
--learning_rate 5e-5 \
--logging_steps 5000 \
--seed 9527 \
--num_train_epochs 10 \
--max_epochs 10 \
--control_modes 5 \
--gradient_accumulation_steps 3 \
--input_trunc_length 512 \
--fp16
```
- CMDP training for D.GPT2 model on Newsroom-b, change `--data_dir` to the path of `cased-cnn-dailymail` if you want to use CNN/DM. 
```
python3 -m torch.distributed.launch \
--master_port=1234 \
--nproc_per_node 4 gpt2_summarization_rl_finetuning.py \
--data_dir ../../datasets/newsroom_guardian_ws_nyt \
--output_dir saved_models/train_rl_distilgpt2_cnn_mode_5_3_bins_batch4_6gpu_fp_16_epoch_20_512_tokens \
--model_type gpt2 \
--model_name_or_path saved_models/train_ml_distilgpt2_cnn_mode_5_3_bins_batch4_6gpu_fp_16_epoch_10_512_tokens/epoch_states/10-epoch \
--tokenizer_name saved_models/train_ml_distilgpt2_cnn_mode_5_3_bins_batch4_6gpu_fp_16_epoch_10_512_tokens \
--cache_dir /data/model_cache \
--do_train \
--evaluate_during_training \
--per_gpu_train_batch_size 12 \
--per_gpu_eval_batch_size 24 \
--learning_rate 1.77e-5 \
--max_output_length 75 \
--seed 9527 \
--num_train_epochs 20 \
--reward_type 9 \
--baseline self \
--constrained_mdp \
--cost_types 22 15 30 \
--cost_thresholds 0.0 0.0 \
--lagrangian_init_val 0.01 \
--logging_steps 1000 \
--control_modes 5 \
--gradient_accumulation_steps 3 \
--fp16 \
--num_freeze_layers 4
```
### Testing
- Decode from D.GPT2 model with abstractiveness bin 1. You should set `--desired_target_number 1` for abstractiveness bin 2 and `--desired_target_number 2` for abstractiveness bin 3.
```
python3 -u gpt2_summarization_prediction.py \
--model_type gpt2 \
--model_name_or_path saved_models/train_rl_distilgpt2_cnn_bertscore_control_mode_5_new_epoch20/epoch_states/17-epoch \
--tokenizer_name saved_models/train_rl_distilgpt2_cnn_bertscore_control_mode_5_new_epoch20 \
--pred_path pred/predict_rl_distilgpt2_cnn_mode_5_new_3_bins_512_tokens_bin_0 \
--data_dir ../../datasets/newsroom_guardian_ws_nyt \
--split test \
--temperature 1.0 \
--batch_size 4 \
--beam_size 5 \
--control_modes 5 \
--desired_target_number 0 \
--seed 9527 \
--input_trunc_length 512
```
- Evaluate BERTScore
```
python evaluate_bert_score.py --decode_dir pred/predict_rl_distilgpt2_cnn_mode_5_new_3_bins_512_tokens_bin_0 --data ../../datasets/newsroom_guardian_ws_nyt --model bert-base-uncased --batch_size 40 --rescale-with-baseline --lang en
```
- Evaluate MoverScore
```
python evaluate_mover_score.py --decode_dir pred/predict_rl_distilgpt2_cnn_mode_5_new_3_bins_512_tokens_bin_0 --data ../../datasets/newsroom_guardian_ws_nyt --batch_size 64
```
- Evaluate Bin % for bin 1 output, you should set `-target_abs_bin 1` for abstractiveness bin 2 and `-target_abs_bin 2` for abstractiveness bin 3. 
```
python output_extractive_fragment_density_stat.py -data_dir ../../datasets/newsroom_guardian_ws_nyt -decode_dir pred/predict_rl_distilgpt2_newsroom_penalty_mode_5_new_3_bins_512_tokens_bin_0_epoch12_dgx -split test -target_abs_bin 0
```

## Model Output
You can download our decoded summaries [here](https://www.dropbox.com/sh/s3h2a5wclmir1zy/AADp5E0KmOzSIYCU792QYWDDa?dl=0)

## Values of learned Lagrangian multipliers
The values of learned Lagrangian multipliers λ changes dynamically during training. In the following tables, we report the learned values of λ of our D.GPT2+CMDP model when the validation reward converges. 

**Length control:**
|                             | length bin constraint | 3-gram repetition constraint |
|-----------------------------|-----------------------|------------------------------|
| Values of learned λ   | 0.3312                | 0.3333                       |

**Entity control:**
|                             | QA constraint | entity repetition constraint | 3-gram repetition constraint |
|-----------------------------|-----------------------|------------------------------|------------------------------|
| Values of learned λ   | 0.1980                | 0.1810                       | 0.1972                       |

**Abstractiveness control:**
|                             | Abstractiveness bin constraint | conjunction constraint | 3-gram repetition constraint |
|-----------------------------|-----------------------|------------------------------|------------------------------|
| Values of learned λ (CNN/DM)   | 0.2842                | 0.1271                       | 0.2952                       |
| Values of learned λ (Newsroom-b)   | 0.4832                | 0.2210                       | 0.4898                       |


## Distribution of length bins in the CNN/DM training set

| Bin           | Range         | % of samples  |
| ------------- |:-------------:| -----:|
| 1      | (0,33] | 10.32 |
| 2      | (33,38] | 9.07 |
| 3      | (38,42] | 9.85 |
| 4      | (42,46] | 8.61 |
| 5      | (46,50] | 8.68 |
| 6      | (50,54] | 9.41 |
| 7      | (54,59] | 10.58 |
| 8      | (59,64] | 8.23 |
| 9      | (64,72] | 8.85 |
| 10      | (72,94] | 15.78 |
