<h1 align="center">
<p>ViCorrection: Vietnamese Spelling Correction
</h1>

## Overview 
A project to correct spelling errors in Vietnamese texts using Seq2Seq and Seq2Seq-Attention models at character-level

More information can read from [training.ipynb](notebooks/Spelling_Correction_Vietnamese_Training.ipynb) or [testing.ipynb](notebooks/Spelling_Correct_Vietnamese_Testing.ipynb)

### Setup 
```bash
pip install -r requirements.txt
```

### Create dataset
Change .csv path and save_path in dataset/create_dataset.py and run:
```bash 
cd dataset
python create_dataset.py
```

### Model (character-level)
##### List of neural models in the repo:

- [```Seq2Seq```](weights/seq2seq_without_att.pth)
- [```Seq2Seq-BahdanauAttention```](weights/seq2seq.pth)

### Training
Edit parameters in params.py file and training:
```bash
python training.py
```

# Performances
| Spell<br>Checker    | CER (%) | Full Sequence Acc (%) |
|----------|----------------------|--------------------------------------|
| ```Seq2Seq``` | 1.34 | 82.3 |
|``` Seq2Seq-Attention``` | 1.12 | 85.7|

