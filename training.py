from models.trainer import Trainer
from params import alphabets
import numpy as np
import os

if not os.path.exists('./checkpoint'):
  os.mkdir('./checkpoint')

if not os.path.exists('./weights'):
  os.mkdir('./weights')

if not os.path.exists('./log'):
  os.mkdir('./log')

def load_dataset(ngrams_path):
    if not os.path.exists(ngrams_path):
        print("Cannot find ngrams path !!!")
    print("Loading dataset...")
    with open(ngrams_path, 'rb') as f:
        list_ngrams = np.load(f)
    print("Num samples of dataset: ", list_ngrams.shape[0])
    print("Loaded dataset!!!")

    return list_ngrams


def training(ngrams_path, resume=False, checkpoint_path=""):
    list_ngrams = load_dataset(ngrams_path)
    trainer = Trainer(alphabets, list_ngram=list_ngrams)

    # Resume training
    if resume:
        trainer.load_checkpoint(checkpoint_path)

    # Training
    trainer.train()


if __name__ == '__main__':
    training(ngrams_path="./data/list_ngrams.npy")


