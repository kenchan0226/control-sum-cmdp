""" Evaluate the baselines ont ROUGE/METEOR"""
""" Adapted from https://github.com/ChenRocks/fast_abs_rl """
import argparse
import json
import os
from os.path import join, exists
import bert_score
import re
import torch


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.ref')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def _read_file(filename):
    # print(dec_fname)
    summary_sent_list_lower = []
    with open(filename) as f:
        for _, l in enumerate(f):
            summary_sent_list_lower.append(l.strip().lower())
    summary_str_lower = ' '.join(summary_sent_list_lower)
    return summary_str_lower


def _construct_list(dec_dir, ref_dir):
    print(dec_dir)
    print(ref_dir)
    n_data = _count_data(ref_dir)
    output_summary_str_list = []
    ref_summary_str_list = []
    for i in range(n_data):
        dec_fname = join(dec_dir, '{}.dec'.format(i))
        output_summary_str_lower = _read_file(dec_fname)
        output_summary_str_list.append(output_summary_str_lower)
        ref_fname = join(ref_dir, '{}.ref'.format(i))
        ref_summary_str_lower = _read_file(ref_fname)
        ref_summary_str_list.append(ref_summary_str_lower)
    return output_summary_str_list, ref_summary_str_list


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser('Calculate BERTScore')
    parser.add_argument('--lang', type=str, default=None,
                        help='two-letter abbreviation of the language (e.g., en) or "en-sci" for scientific text')
    parser.add_argument('-m', '--model', default=None,
                        help='BERT model name (default: bert-base-uncased) or path to a pretrain model')
    parser.add_argument('-l', '--num_layers', type=int, default=None, help='use first N layer in BERT (default: 8)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--nthreads', type=int, default=4, help='number of cpu workers (default: 4)')
    parser.add_argument('--idf', action='store_true', help='BERT Score with IDF scaling')
    parser.add_argument('--rescale-with-baseline', action='store_true', help='Rescaling the numerical score with precomputed baselines')
    #parser.add_argument('-s', '--seg_level', action='store_true', help='show individual score of each pair')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser.add_argument('--decode_dir', action='store', required=True, help='directory of decoded summaries')
    parser.add_argument('--data', action='store', required=True, help='directory of decoded summaries')

    args = parser.parse_args()

    dec_dir = join(args.decode_dir, 'output')
    with open(join(args.decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    ref_dir = join(args.data, 'refs', split)
    print(ref_dir)
    assert exists(ref_dir)

    output_summary_str_list, ref_summary_str_list = _construct_list(dec_dir, ref_dir)
    all_preds, hash_code = bert_score.score(cands=output_summary_str_list, refs=ref_summary_str_list, model_type=args.model, num_layers=args.num_layers,
                                            verbose=args.verbose, idf=args.idf, batch_size=args.batch_size,
                                            lang=args.lang, return_hash=True,
                                            rescale_with_baseline=args.rescale_with_baseline)
    avg_scores = [s.mean(dim=0) for s in all_preds]
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    msg = hash_code + \
        f' R: {R:.6f} P: {P:.6f} F1: {F1:.6f}'
    print(msg)
    """
    if args.seg_level:
        ps, rs, fs = all_preds
        for p, r, f in zip(ps, rs, fs):
            print('{:.6f}\t{:.6f}\t{:.6f}'.format(p, r, f))
    """

    f1_all = all_preds[2]
    f1_all_list = f1_all.cpu().tolist()
    with open(join(args.decode_dir, 'bertscore.txt'), 'w') as f:
        for f1 in f1_all_list:
            f.write("{:.6f}\n".format(f1))
    print("Finish!")


if __name__ == "__main__":
    main()
