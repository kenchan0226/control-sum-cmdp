import pickle as pkl
import argparse
from os.path import join
import numpy as np


LEXICON_PATH = "/research/king3/hpchan/projects/seq-gen-mdp/word_complexity_lexicon/word_to_complexity.pkl"
VOCAB_PATH = "/research/king3/hpchan/datasets/v3.7_newsroom/vocab_cnt.pkl"

def main():
    with open(LEXICON_PATH, 'rb') as f:
        word_to_complexity = pkl.load(f)

    with open(VOCAB_PATH, 'rb') as f:
        vocab_cnt = pkl.load(f)

    complexity_all = []

    for w, cnt in vocab_cnt.most_common(50000):
        complexity = word_to_complexity[w]
        complexity_all.append(complexity)

    complexity_array = np.array(complexity_all)
    hist, bins = np.histogram(complexity_array, bins=[0,1,2,3,4,5,6,7], density=False)
    print(hist)
    print(bins)

if __name__ == "__main__":
    main()
