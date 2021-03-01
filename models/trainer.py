from torch.optim import Adam, SGD, AdamW
import torch
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from dataset.vocab import Vocab
from dataset.add_noise import SynthesizeData
from params import *
from models.seq2seq import Seq2Seq
from models.seq2seq_without_attention import Seq2Seq_WithoutAtt
from utils.logger import Logger
from dataset.autocorrect_dataset import AutoCorrectDataset
from models.loss import LabelSmoothingLoss
from utils.utils import translate, translate_beam_search, batch_translate_beam_search
from utils.metrics import compute_accuracy


class Trainer():
    def __init__(self, alphabets_, list_ngram):

        self.vocab = Vocab(alphabets_)
        self.synthesizer = SynthesizeData(vocab_path="")
        self.list_ngrams_train, self.list_ngrams_valid = self.train_test_split(list_ngram, test_size=0.1)
        print("Loaded data!!!")
        print("Total training samples: ", len(self.list_ngrams_train))
        print("Total valid samples: ", len(self.list_ngrams_valid))

        INPUT_DIM = self.vocab.__len__()
        OUTPUT_DIM = self.vocab.__len__()

        self.device = DEVICE
        self.num_iters = NUM_ITERS
        self.beamsearch = BEAM_SEARCH

        self.batch_size = BATCH_SIZE
        self.print_every = PRINT_PER_ITER
        self.valid_every = VALID_PER_ITER

        self.checkpoint = CHECKPOINT
        self.export_weights = EXPORT
        self.metrics = MAX_SAMPLE_VALID
        logger = LOG

        if logger:
            self.logger = Logger(logger)

        self.iter = 0

        self.model = Seq2Seq(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, encoder_embbeded=ENC_EMB_DIM,
                             decoder_embedded=DEC_EMB_DIM,
                             encoder_hidden=ENC_HID_DIM, decoder_hidden=DEC_HID_DIM, encoder_dropout=ENC_DROPOUT,
                             decoder_dropout=DEC_DROPOUT)

        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, pct_start=PCT_START, max_lr=MAX_LR)

        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

        self.train_gen = self.data_gen(self.list_ngrams_train, self.synthesizer, self.vocab, is_train=True)
        self.valid_gen = self.data_gen(self.list_ngrams_valid, self.synthesizer, self.vocab, is_train=False)

        self.train_losses = []

        # to device
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train_test_split(self, list_phrases, test_size=0.1):
        list_phrases = list_phrases
        train_idx = int(len(list_phrases) * (1 - test_size))
        list_phrases_train = list_phrases[:train_idx]
        list_phrases_valid = list_phrases[train_idx:]
        return list_phrases_train, list_phrases_valid

    def data_gen(self, list_ngrams_np, synthesizer, vocab, is_train=True):
        dataset = AutoCorrectDataset(list_ngrams_np, transform_noise=synthesizer, vocab=vocab, maxlen=MAXLEN)

        shuffle = True if is_train else False
        gen = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            drop_last=False)

        return gen

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        src, tgt = batch['src'], batch['tgt']
        src, tgt = src.transpose(1, 0), tgt.transpose(1, 0)  # batch x src_len -> src_len x batch

        outputs = self.model(src, tgt)  # src : src_len x B, outpus : B x tgt_len x vocab

        #        loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))
        outputs = outputs.view(-1, outputs.size(2))  # flatten(0, 1)

        tgt_output = tgt.transpose(0, 1).reshape(-1)  # flatten()   # tgt: tgt_len xB , need convert to B x tgt_len

        loss = self.criterion(outputs, tgt_output)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        self.scheduler.step()

        loss_item = loss.item()

        return loss_item

    def train(self):
        print("Begin training from iter: ", self.iter)
        total_loss = 0

        total_loader_time = 0
        total_gpu_time = 0
        best_acc = -1

        data_iter = iter(self.train_gen)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(
                    self.iter,
                    total_loss / self.print_every, self.optimizer.param_groups[0]['lr'],
                    total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info)
                self.logger.log(info)

            if self.iter % self.valid_every == 0:
                val_loss, preds, actuals, inp_sents = self.validate()
                acc_full_seq, acc_per_char, cer = self.precision(self.metrics)

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f} - CER: {:.4f} '.format(
                    self.iter, val_loss, acc_full_seq, acc_per_char, cer)
                print(info)
                print("--- Sentence predict ---")
                for pred, inp, label in zip(preds, inp_sents, actuals):
                    infor_predict = 'Pred: {} - Inp: {} - Label: {}'.format(pred, inp, label)
                    print(infor_predict)
                    self.logger.log(infor_predict)
                self.logger.log(info)

                if acc_full_seq > best_acc:
                    self.save_weights(self.export_weights)
                    best_acc = acc_full_seq
                self.save_checkpoint(self.checkpoint)

    def validate(self):
        self.model.eval()

        total_loss = []
        max_step = self.metrics / self.batch_size
        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen):
                batch = self.batch_to_device(batch)
                src, tgt = batch['src'], batch['tgt']
                src, tgt = src.transpose(1, 0), tgt.transpose(1, 0)

                outputs = self.model(src, tgt, 0)  # turn off teaching force

                outputs = outputs.flatten(0, 1)
                tgt_output = tgt.flatten()
                loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())

                preds, actuals, inp_sents, probs = self.predict(5)

                del outputs
                del loss
                if step > max_step:
                    break

        total_loss = np.mean(total_loss)
        self.model.train()

        return total_loss, preds[:3], actuals[:3], inp_sents[:3]

    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        inp_sents = []

        for batch in self.valid_gen:
            batch = self.batch_to_device(batch)

            if self.beamsearch:
                translated_sentence = batch_translate_beam_search(batch['src'], self.model)
                prob = None
            else:
                translated_sentence, prob = translate(batch['src'], self.model)

            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt'].tolist())
            inp_sent = self.vocab.batch_decode(batch['src'].tolist())

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            inp_sents.extend(inp_sent)

            if sample is not None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, inp_sents, prob

    def precision(self, sample=None):

        pred_sents, actual_sents, _, _ = self.predict(sample=sample)

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')
        cer = compute_accuracy(actual_sents, pred_sents, mode='CER')

        return acc_full_seq, acc_per_char, cer

    def visualize_prediction(self, sample=16, errorcase=False, fontname='serif', fontsize=16):

        pred_sents, actual_sents, img_files, probs = self.predict(sample)

        if errorcase:
            wrongs = []
            for i in range(len(img_files)):
                if pred_sents[i] != actual_sents[i]:
                    wrongs.append(i)

            pred_sents = [pred_sents[i] for i in wrongs]
            actual_sents = [actual_sents[i] for i in wrongs]
            img_files = [img_files[i] for i in wrongs]
            probs = [probs[i] for i in wrongs]

        img_files = img_files[:sample]

        fontdict = {
            'family': fontname,
            'size': fontsize
        }


    def visualize_dataset(self, sample=16, fontname='serif'):
        n = 0
        for batch in self.train_gen:
            for i in range(self.batch_size):
                img = batch['img'][i].numpy().transpose(1, 2, 0)
                sent = self.vocab.decode(batch['tgt_input'].T[i].tolist())

                n += 1
                if n >= sample:
                    return

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.iter = checkpoint['iter']

        self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'iter': self.iter, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses,
                 'scheduler': self.scheduler.state_dict()}

        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print(
                    '{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), filename)

    def batch_to_device(self, batch):

        src = batch['src'].to(self.device, non_blocking=True)
        tgt = batch['tgt'].to(self.device, non_blocking=True)

        batch = {
            'src': src,
            'tgt': tgt
        }

        return batch