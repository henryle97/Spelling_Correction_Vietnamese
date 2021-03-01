import torch
import numpy as np
from params import PERCENT_NOISE

class AutoCorrectDataset(torch.utils.data.Dataset):
    def __init__(self, list_ngram, transform_noise, vocab, maxlen):
        self.list_ngram = list_ngram
        self.transform_noise = transform_noise
        self.vocab = vocab
        self.maxlen = maxlen

    def __getitem__(self, idx):
        correct_sent = self.list_ngram[idx]
        noise_sent = self.transform_noise.add_noise(correct_sent, percent_err=PERCENT_NOISE)
        # print(correct_sent)
        # print(noise_sent)
        correct_sent_idxs = self.vocab.encode(correct_sent)
        noise_sent_idxs = self.vocab.encode(noise_sent)

        src_len = len(noise_sent_idxs)
        if self.maxlen - src_len < 0:
            noise_sent_idxs = noise_sent_idxs[:self.maxlen]
            src_len = len(noise_sent_idxs)
            print("Over length in src")
        src = np.concatenate((
            noise_sent_idxs,
            np.zeros(self.maxlen - src_len, dtype=np.int32)))

        tgt_len = len(correct_sent_idxs)
        if self.maxlen - tgt_len < 0:
            correct_sent_idxs = correct_sent_idxs[:self.maxlen]
            tgt_len = len(correct_sent_idxs)
            print("Over length in target")
        tgt = np.concatenate((
            correct_sent_idxs,
            np.zeros(self.maxlen - tgt_len, dtype=np.int32)))

        rs = {
            'src': torch.LongTensor(src),
            'tgt': torch.LongTensor(tgt),
        }

        return rs

    def __len__(self):
        return len(self.list_ngram)

