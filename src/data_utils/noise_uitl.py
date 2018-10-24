import itertools
from collections import Counter
import pandas as pd
import numpy as np

import hyperparams as hp

np.random.seed(0)

def generate_random_noise(sents, noise_rate, vocab_size=8000):
    tokens_count = Counter(itertools.chain(*sents))
    vocab = [token[0]
             for token in tokens_count.most_common(vocab_size-1)]
    vocab += ['<OOV>']
    i = 0
    while i < len(sents):
        num_noise_words = min(int(len(sents[i]) * noise_rate), hp.FORCED_SEQ_LEN-len(sents[i]))
        for _ in range(num_noise_words):
            noise_word = vocab[np.random.randint(8000)]
            noise_loc = np.random.randint(len(sents[i]))
            sents[i].insert(noise_loc, noise_word)
        i += 1
    return sents
