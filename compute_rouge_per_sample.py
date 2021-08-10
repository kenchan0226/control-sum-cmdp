""" Evaluate the baselines ont ROUGE/METEOR"""
""" Adapted from https://github.com/ChenRocks/fast_abs_rl """
import argparse
import json
import os
from os.path import join, exists
import re
from utils.evaluate import eval_rouge
from shutil import copyfile
from tqdm import tqdm


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.dec')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def main(args):
    dec_dir = join(args.decode_dir, 'output')
    with open(join(args.decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    ref_dir = join(args.data, 'refs', split)
    assert exists(ref_dir)

    # construct tmp folder
    tmp_dec_dir = join(args.decode_dir, 'output_tmp')
    os.makedirs(tmp_dec_dir)
    tmp_ref_dir = join(args.data, 'refs', args.tmp_ref_dir_name)
    #tmp_ref_dir = join(args.data, 'refs', split + '_tmp')
    os.makedirs(tmp_ref_dir)
    if args.multi_ref:
        tmp_ref_filename_1 = join(tmp_ref_dir, "A.0.ref")
        tmp_ref_filename_2 = join(tmp_ref_dir, "B.0.ref")
    else:
        tmp_ref_filename = join(tmp_ref_dir, "0.ref")
    tmp_dec_filename = join(tmp_dec_dir, "0.dec")
    n_data = _count_data(dec_dir)

    dec_pattern = r'(\d+).dec'
    if args.multi_ref:
        ref_pattern = '[A-Z].#ID#.ref'
        print("ref_pattern")
    else:
        ref_pattern = '#ID#.ref'
    rouge_1_all_list = []
    rouge_2_all_list = []
    rouge_l_all_list = []
    for i in tqdm(range(n_data)):
        # copy ref and dec
        if args.multi_ref:
            ref_file_to_copy_1 = join(ref_dir, "A.{}.ref".format(i))
            copyfile(ref_file_to_copy_1, tmp_ref_filename_1)
            ref_file_to_copy_2 = join(ref_dir, "B.{}.ref".format(i))
            if os.path.isfile(ref_file_to_copy_2):
                copyfile(ref_file_to_copy_2, tmp_ref_filename_2)
                ref_b = True
            else:
                ref_b = False
                print("without b")
        else:
            ref_file_to_copy = join(ref_dir, "{}.ref".format(i))
            copyfile(ref_file_to_copy, tmp_ref_filename)
        #print(ref_file_to_copy)
        #print(tmp_ref_filename)
        dec_file_to_copy = join(dec_dir, "{}.dec".format(i))
        copyfile(dec_file_to_copy, tmp_dec_filename)
        #print(dec_file_to_copy)
        #print(tmp_dec_filename)
        output = eval_rouge(dec_pattern, tmp_dec_dir, ref_pattern, tmp_ref_dir, n_words=args.n_words, n_bytes=args.n_bytes,
                            cmd='-n 2 -m')
        #print(output)
        output_lines = output.split("\n")
        rouge_1 = output_lines[3].split(" ")[3]
        rouge_2 = output_lines[7].split(" ")[3]
        rouge_l = output_lines[11].split(" ")[3]

        rouge_1_all_list.append(rouge_1)
        rouge_2_all_list.append(rouge_2)
        rouge_l_all_list.append(rouge_l)

        # delete tmp file
        if args.multi_ref:
            os.remove(tmp_ref_filename_1)
            if ref_b:
                os.remove(tmp_ref_filename_2)
        else:
            os.remove(tmp_ref_filename)
        os.remove(tmp_dec_filename)

    os.rmdir(tmp_ref_dir)
    os.rmdir(tmp_dec_dir)
    with open(join(args.decode_dir, 'r1.txt'), 'w') as f:
        f.write("\n".join(rouge_1_all_list))
    with open(join(args.decode_dir, 'r2.txt'), 'w') as f:
        f.write("\n".join(rouge_2_all_list))
    with open(join(args.decode_dir, 'rl.txt'), 'w') as f:
        f.write("\n".join(rouge_l_all_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')

    # choose metric to evaluate
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
    parser.add_argument('-tmp_ref_dir_name', action='store', default="test_tmp",
                        help='directory of decoded summaries')

    args = parser.parse_args()
    main(args)
