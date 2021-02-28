import re
import nltk
from collections import Counter
from dataset.vocab import Vocab
import torch
from params import *
from models.seq2seq import Seq2Seq
from utils.utils import *
from dataset.add_noise import SynthesizeData
from models.seq2seq_without_attention import Seq2Seq_WithoutAtt

class Predictor:
    def __init__(self, weight_path, have_att=False):
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        ENC_HID_DIM = 512
        DEC_HID_DIM = 512
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5
        MAX_LEN = 46
        self.maxlen = MAX_LEN
        self.vocab = Vocab(alphabets)

        INPUT_DIM = self.vocab.__len__()
        OUTPUT_DIM = self.vocab.__len__()

        if have_att:
            self.model = Seq2Seq(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, encoder_embbeded=ENC_EMB_DIM,
                                 decoder_embedded=DEC_EMB_DIM,
                                 encoder_hidden=ENC_HID_DIM, decoder_hidden=DEC_HID_DIM, encoder_dropout=ENC_DROPOUT,
                                 decoder_dropout=DEC_DROPOUT)
        else:
            self.model = Seq2Seq_WithoutAtt(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, encoder_embbeded=ENC_EMB_DIM,
                                 decoder_embedded=DEC_EMB_DIM,
                                 encoder_hidden=ENC_HID_DIM, decoder_hidden=DEC_HID_DIM, encoder_dropout=ENC_DROPOUT,
                                 decoder_dropout=DEC_DROPOUT)

        self.load_weights(weight_path)
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model.to('cuda')
        else:
            self.device = "cpu"

        print("Device: ", self.device)
        print("Loaded model")

    def predict_ngram(self, ngram, beamsearch=False):
        '''
          Denoise for ngram
          ngram: text
        '''
        src = self.preprocessing(ngram)
        src = src.unsqueeze(0)
        src = src.to(self.device)

        if beamsearch:
            translated_sentence = batch_translate_beam_search(src, self.model)
            prob = None
        else:
            translated_sentence, prob = translate(src, self.model)
        # print(translated_sentence)
        pred_sent = self.vocab.decode(translated_sentence.tolist()[0])

        return pred_sent

    def spelling_correct(self, sentence):
        # Remove characters that out of vocab
        sentence = re.sub(
            r'[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ ]',
            "", sentence)

        # Extract pharses
        phrases, phrases_all, index_sent_dict = self.extract_phrases(sentence)

        correct_phrases = []
        for phrase in phrases:
            ngrams = list(self.gen_ngrams(phrase, n=NGRAM))
            correct_ngram_str_array = []
            for ngram_list in ngrams:
                ngram_str = " ".join(ngram_list)

                correct_ngram_str = self.predict_ngram(ngram_str)
                correct_ngram_str_array.append(correct_ngram_str)
            correct_phrase = self.reconstruct_from_ngrams(correct_ngram_str_array)
            correct_phrases.append(correct_phrase)
        correct_sentence = self.decode_phrases(correct_phrases, phrases_all, index_sent_dict)
        return correct_sentence

    def reconstruct_from_ngrams(self, predicted_ngrams):
        '''
        predicted_ngrams: list of ngram_str
        '''

        candidates = [Counter() for _ in range(len(predicted_ngrams) + NGRAM - 1)]
        for nid, ngram in (enumerate(predicted_ngrams)):
            tokens = re.split(r' +', ngram)
            for wid, word in enumerate(tokens):
                candidates[nid + wid].update([word])
        # print(candidates)
        output = ' '.join(c.most_common(1)[0][0] for c in candidates if len(c) != 0)
        return output

    def extract_phrases(self, text):
        pattern = r'\w[\w ]*|\s\W+|\W+'

        phrases_all = re.findall(pattern, text)

        index_sent_dict = {}
        phrases_str = []
        for ind, phrase in enumerate(phrases_all):
            if not re.match(r'[!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~]', phrase.strip()):
                phrases_str.append(phrase.strip())
                index_sent_dict[ind] = phrase

        return phrases_str, phrases_all, index_sent_dict

    def decode_phrases(self, correct_phrases, phrases, index_sent_dict):
        # correct_phrases = ['lê văn', 'Hoàng', 'Hehe', 'g']
        sentence_correct = phrases.copy()
        for i, idx_sent in enumerate(index_sent_dict.keys()):
            sentence_correct[idx_sent] = correct_phrases[i]

        # print(sentence_correct)
        return "".join(sentence_correct)

    def preprocessing(self, sentence):

        # Encode characters
        noise_sent_idxs = self.vocab.encode(sentence)

        # Padding to MAXLEN
        src_len = len(noise_sent_idxs)
        if self.maxlen - src_len < 0:
            noise_sent_idxs = noise_sent_idxs[:self.maxlen]
            src_len = len(noise_sent_idxs)
            print("Over length in src")
        src = np.concatenate((
            noise_sent_idxs,
            np.zeros(self.maxlen - src_len, dtype=np.int32)))

        return torch.LongTensor(src)

    def gen_ngrams(self, sent, n=5):
        tokens = sent.split()

        if len(tokens) < n:
            return [tokens]

        return nltk.ngrams(sent.split(), n)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device('cpu'))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print(
                    '{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

if __name__ == "__main__":
    predictor = Predictor(weight_path='weights/seq2seq.pth')
    synther = SynthesizeData()
    noise_sent = synther.add_noise("tôi là sinh viên", percent_err=0.15)

    print("Noise: ", noise_sent)
    correct = predictor.spelling_correct(noise_sent)
    print("Correct: ", correct)