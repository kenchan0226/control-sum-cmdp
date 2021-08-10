""" Evaluate the baselines ont ROUGE/METEOR"""
""" Adapted from https://github.com/ChenRocks/fast_abs_rl """
import argparse
import json
import os
from os.path import join, exists

from utils.evaluate import eval_meteor, eval_rouge


def main(args):
    dec_dir = join(args.decode_dir, 'output')
    with open(join(args.decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    ref_dir = join(args.data, 'refs', split)
    assert exists(ref_dir)

    if args.rouge:
        dec_pattern = r'(\d+).dec'
        if args.multi_ref:
            ref_pattern = '[A-Z].#ID#.ref'
        else:
            ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir, n_words=args.n_words, n_bytes=args.n_bytes)
        metric = 'rouge'
    else:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'
    print(output)
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('-rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('-meteor', action='store_true',
                            help='METEOR evaluation')

    parser.add_argument('-decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('-data', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('-n_words', type=int, action='store', default=-1,
                        help='Only use the first n words in the system/peer summary for the evaluation.')
    parser.add_argument('-n_bytes', type=int, action='store', default=-1,
                        help='Only use the first n bytes in the system/peer summary for the evaluation.')
    parser.add_argument('-multi_ref', action='store_true',
                            help='Use multiple references')

    args = parser.parse_args()
    main(args)
