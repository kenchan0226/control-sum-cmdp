""" ROUGE utils"""
""" Adapted from https://github.com/ChenRocks/fast_abs_rl """

import os
import threading
import subprocess as sp
from collections import Counter, deque

from cytoolz import concat, curry


def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    return ngrams

def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count

@curry
def compute_weighted_rouge_1_2(output, reference, rouge_1_weight=0.5, mode='f'):
    rouge_1 = compute_rouge_n(output, reference, n=1, mode=mode)
    rouge_2 = compute_rouge_n(output, reference, n=2, mode=mode)
    return rouge_1_weight * rouge_1 + (1-rouge_1_weight) * rouge_2

@curry
def compute_rouge_n(output, reference, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


@curry
def compute_rouge_n_zipped(args, n=1, mode='f'):
    """ compute ROUGE-N for a single pair of summary and reference"""
    assert mode in list('fpr')  # F-1, precision, recall
    output, reference = args
    match = _n_gram_match(reference, output, n)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp

def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]

@curry
def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _weight_word_list(word_list, important_word_list):
    total_weight = 0.0
    for word in word_list:
        if word in important_word_list:
            weight = 1.0
        else:
            weight = 0.5
        total_weight += weight
    return total_weight


def _weight_word(word, important_word_list):
    if word in important_word_list:
        return 1.0
    else:
        return 0.5


@curry
def compute_weighted_rouge_l(output, reference, important_word_list, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    m = len(output)
    n = len(reference)
    lcs_word_list = _lcs(output, reference)
    if len(lcs_word_list) == 0:
        score = 0.0
    else:
        # weight
        lcs_weighted = _weight_word_list(lcs_word_list, important_word_list)
        precision = lcs_weighted / _weight_word_list(output, important_word_list)
        recall = lcs_weighted / _weight_word_list(reference, important_word_list)
        #precision = lcs_weighted / len(output)
        #recall = lcs_weighted / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _lcs(a, b):
    """ compute the longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    i = len(a)
    j = len(b)
    lcs = deque()
    while (i > 0 and j > 0):
        if a[i-1] == b[j-1]:
            lcs.appendleft(a[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    assert len(lcs) == dp[-1][-1]
    return lcs

@curry
def compute_rouge_l_summ(summs, refs, mode='f'):
    """ summary level ROUGE-L"""
    assert mode in list('fpr')  # F-1, precision, recall
    tot_hit = 0
    ref_cnt = Counter(concat(refs))
    summ_cnt = Counter(concat(summs))
    for ref in refs:
        for summ in summs:
            lcs = _lcs(summ, ref)
            for gram in lcs:
                if ref_cnt[gram] > 0 and summ_cnt[gram] > 0:
                    tot_hit += 1
                ref_cnt[gram] -= 1
                summ_cnt[gram] -= 1
    if tot_hit == 0:
        score = 0.0
    else:
        precision = tot_hit / sum((len(s) for s in summs))
        recall = tot_hit / sum((len(r) for r in refs))
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


@curry
def compute_weighted_rouge_l_summ(summs, refs, important_word_list, mode='f'):
    """ summary level ROUGE-L"""
    assert mode in list('fpr')  # F-1, precision, recall
    tot_hit = 0
    ref_cnt = Counter(concat(refs))
    summ_cnt = Counter(concat(summs))
    for ref in refs:
        for summ in summs:
            lcs = _lcs(summ, ref)
            for gram in lcs:
                if ref_cnt[gram] > 0 and summ_cnt[gram] > 0:
                    #tot_hit += 1
                    tot_hit += _weight_word(gram, important_word_list)
                ref_cnt[gram] -= 1
                summ_cnt[gram] -= 1
    if tot_hit == 0:
        score = 0.0
    else:
        #precision = tot_hit / sum((len(s) for s in summs))
        #recall = tot_hit / sum((len(r) for r in refs))
        precision = tot_hit / sum((_weight_word_list(s, important_word_list) for s in summs))
        recall = tot_hit / sum((_weight_word_list(r, important_word_list) for r in refs))
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None
class Meteor(object):
    def __init__(self):
        assert _METEOR_PATH is not None
        cmd = 'java -Xmx2G -jar {} - - -l en -norm -stdio'.format(_METEOR_PATH)
        self._meteor_proc = sp.Popen(
            cmd.split(),
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE,
            universal_newlines=True, encoding='utf-8', bufsize=1
        )
        self._lock = threading.Lock()

    def __call__(self, summ, ref):
        self._lock.acquire()
        score_line = 'SCORE ||| {} ||| {}\n'.format(
            ' '.join(ref), ' '.join(summ))
        self._meteor_proc.stdin.write(score_line)
        stats = self._meteor_proc.stdout.readline().strip()
        eval_line = 'EVAL ||| {}\n'.format(stats)
        self._meteor_proc.stdin.write(eval_line)
        score = float(self._meteor_proc.stdout.readline().strip())
        self._lock.release()
        return score

    def __del__(self):
        self._lock.acquire()
        self._meteor_proc.stdin.close()
        self._meteor_proc.kill()
        self._meteor_proc.wait()
        self._lock.release()


if __name__ == "__main__":
    important_word_list = ["man", "united", "liverpool"]
    #important_word_list = ["deal"]
    output = ["louis van gaal is close to delivering his first-season aim of returning man united into champions league. united ???s win over aston villa took them third ,",
              "eight points ahead of fifth-placed liverpool in the table ."]
    reference = ["man united have an eight-point cushion from fifth-place liverpool .",
                 "van gaal looks likely to deliver on his promise of top four finish .",
                 "but the dutchman has a three-year vision mapped out ."]
    output_word_list = [o.split(" ") for o in output]
    reference_word_list = [r.split(" ") for r in reference]
    rouge_l_weight = compute_weighted_rouge_l_summ(output_word_list, reference_word_list, important_word_list, mode='f')
    rouge_l = compute_rouge_l_summ(output_word_list, reference_word_list, mode='f')
    print(rouge_l_weight)
    print(rouge_l)
