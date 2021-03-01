import nltk
import pandas as pd
from tqdm import tqdm
import re
import itertools
import sys
sys.path.append("..")
from params import MAXLEN, NGRAM
import numpy as np

'''
structure of csv:

index   |      sentence
  0     |   Oracle Application Server được ưa chuộng  
  1     |   Hãng phần mềm doanh nghiệp hàng đầu thế giới 

'''


class CreateDataset():
    def __init__(self, csv_path='../data/samples_sentence.csv', save_path="../data/list_ngrams.npy"):
        self.csv_path = csv_path
        self.alphabets_regex = '^[aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ ]'
        self.save_path = save_path


    def processing(self):
        # read csv
        df = pd.read_csv(self.csv_path)

        # remove characters that out of vocab
        df['sentence'] = df['sentence'].apply(self.preprocessing_data)

        # extract phrases

        phrases = itertools.chain.from_iterable(self.extract_phrases(text) for text in df['sentence'])
        phrases = [p.strip() for p in phrases if len(p.split()) > 1]

        # gen ngrams
        list_ngrams = []
        for p in tqdm(phrases):
            if not re.match(self.alphabets_regex, p.lower()):
                continue
            if len(phrases) == 0:
                continue

            for ngr in self.gen_ngrams(p, NGRAM):
                if len(" ".join(ngr)) < MAXLEN:
                    list_ngrams.append(" ".join(ngr))
        print("DONE extract ngrams, total ngrams: ", len(list_ngrams))

        # save ngrams
        self.save_ngrams(list_ngrams, save_path=self.save_path)

        print("Done create dataset - ngrams")

    def preprocessing_data(self, row):
        processed = re.sub(
            r'[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ ]',
            "", row)
        return processed

    def extract_phrases(self, text):
        return re.findall(r'\w[\w ]+', text)

    def gen_ngrams(self, text, n=5):
        tokens = text.split()

        if len(tokens) < n:
            return [tokens]

        return nltk.ngrams(text.split(), n)

    def save_ngrams(self, list_ngrams, save_path='ngrams_list.npy'):
        with open(save_path, 'wb') as f:
            np.save(f, list_ngrams)
        print("Saved dataset - ngrams")


if __name__ == "__main__":

    creater = CreateDataset(csv_path='../data/samples_sentence.csv', save_path="../data/list_ngrams.npy")
    creater.processing()
