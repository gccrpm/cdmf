import re
import logging
import itertools
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

def split_train_dev_eval_ratings(train_base_path, other_path):
    dtype = {'userId':np.int32, 'movieId':np.int32, 'rating':np.float, 'timestamp':np.str}
    df_train_base_ratings = pd.read_csv(train_base_path, header=0, dtype=dtype)
    df_other_ratings = pd.read_csv(other_path, header=0, dtype=dtype)

    df_train_ratings = df_other_ratings.sample(frac=0.8, axis=0, random_state=0)
    df_train_ratings = pd.concat([df_train_ratings, df_train_base_ratings], axis=0)
    dev_eval_indices = set(df_other_ratings.index).difference(set(df_train_ratings.index))
    df_dev_eval_ratings = df_other_ratings.loc[dev_eval_indices, :]
    flag = len(dev_eval_indices) // 2

    df_dev_ratings = df_dev_eval_ratings.iloc[:flag, :]
    df_eval_ratings = df_dev_eval_ratings.iloc[flag:, :]

    # save to csv file
    df_train_ratings.to_csv('train_ratings.csv', index=False)
    df_dev_ratings.to_csv('dev_ratings.csv', index=False)
    df_eval_ratings.to_csv('eval_ratings.csv', index=False)

def clean_str(s):
    s = re.sub(r"<[^>]*>", '', s)
    emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", s)
    s += ''.join(emoticons)
    s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
    s = re.sub(r" : ", ":", s)
    s = re.sub(r"\\n", "", s)
    return s.strip().lower()


def pad_sentence(sentences, forced_seq_len=None, padding_word='<PAD>'):
    if forced_seq_len is None:
        # forced_seq_len = max length of all sentences
        forced_seq_len = max([len(sent) for sent in sentences])
        print('max sentences length is {}'.format(forced_seq_len))
    padded_sentences = []
    count_cut_off = 0
    for sent in sentences:
        if len(sent) < forced_seq_len:
            sent.extend([padding_word] * (forced_seq_len - len(sent)))
            padded_sent = sent
        elif len(sent) > forced_seq_len:
            count_cut_off += 1
            padded_sent = sent[:forced_seq_len]
        padded_sentences.append(padded_sent)
    logging.info('Because the length of the sentence is larger the self.forced_seq_len, '
                 'so {} sentences need to be cut off.'.format(count_cut_off))
    return padded_sentences


def build_vocab(sentences, vocab_size=None):
    tokens_count = Counter(itertools.chain(*sentences))
    if vocab_size:
        vocab = [token[0]
                 for token in tokens_count.most_common(vocab_size-1)]
    else:
        vocab = [token[0] for token in tokens_count.most_common()]
        vocab_size = len(vocab)
    vocab += ['<OOV>']  # out of vocablary
    vocab_size += 1
    token2id = {token: i for i, token in enumerate(vocab)}

    return token2id, vocab, vocab_size
