import os

# Loss 1 : Smoothing Loss
import torch
from torch import nn
from torch.nn.functional import log_softmax, softmax
import Levenshtein as Lev
import numpy as np
from models.beam import Beam

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


#  self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)
def cer_calculate(s1, s2, no_spaces=False):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    if no_spaces:
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return min(len(s1), Lev.distance(s1, s2)) / len(s1)


def compute_accuracy(ground_truth, predictions, mode='full_sequence'):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :param mode: if 'per_char' is selected then
                 single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
                 if 'full_sequence' is selected then
                 single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
    :return: avg_label_accuracy
    """
    avg_accuracy = 0

    if mode == 'per_char':

        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == 'full_sequence':
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if prediction == label:
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0
    elif mode == 'CER':
        cer_list = []
        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            cer = cer_calculate(label, prediction)
            cer_list.append(cer)
        avg_accuracy = np.mean(np.array(cer_list).astype(np.float32), axis=0)
    else:
        raise NotImplementedError('Other accuracy compute mode has not been implemented')

    return avg_accuracy


class Logger():
    def __init__(self, fname):
        path, _ = os.path.split(fname)
        os.makedirs(path, exist_ok=True)

        self.logger = open(fname, 'w')

    def log(self, string):
        self.logger.write(string + '\n')
        self.logger.flush()

    def close(self):
        self.logger.close()


def translate(src, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: Bxsrc_len"
    model.eval()
    device = src.device

    with torch.no_grad():
        # src = model.cnn(img)
        src = src.transpose(1, 0)  # src_len x B
        memory = model.forward_encoder(src)

        translated_sentence = [[sos_token] * src.shape[1]]
        char_probs = [[1] * src.shape[1]]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)

            #            output = model(img, tgt_inp, tgt_key_padding_mask=None)
            #            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

    return translated_sentence, char_probs


def batch_translate_beam_search(src, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # src: BxCxHxW
    model.eval()
    device = src.device
    sents = []
    src = src.transpose(1, 0)  # C x B

    with torch.no_grad():
        # src = model.cnn(img)
        # print(src.shap)
        memories = model.forward_encoder(src)
        for i in range(src.size(0)):
            #            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.get_memory(memories, i)
            sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents


def translate_beam_search(src, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()
    device = src.device
    src = src.transpose(1, 0)
    with torch.no_grad():
        # src = model.cnn(img)
        memory = model.forward_encoder(src)  # TxNxE
        sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)

    return sent


def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # memory: Tx1xE
    model.eval()

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token,
                end_token_id=eos_token)

    with torch.no_grad():
        #        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):

            tgt_inp = beam.get_current_state().transpose(0, 1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

    return [1] + [int(i) for i in hypothesises[0][:-1]]