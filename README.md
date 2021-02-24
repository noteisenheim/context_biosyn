# BioSyn with Context Understanding

A research project in the field of biomedical entity normalization.
As the base model, BioSyn is used. ([Code](https://github.com/dmis-lab/BioSyn), [paper](https://arxiv.org/abs/2005.00239))
For the context understanding, S-RoBERTa is used ([Code](https://github.com/UKPLab/sentence-transformers), [paper](https://arxiv.org/abs/1908.10084))

## Prerequisites

### Requirements
Python 3.8 used

```
pip install -r requirements.txt
```

### Datasets

For the training and performance evaluation, **ncbi-disease** dataset was used. It can be downloaded [here](https://drive.google.com/open?id=1nqTQba0IcJiXUal7fx3s-KUFRCfMPpaj).
Unpacked dataset had to be placed in `datasets/ncbi-disease` folder for the default setup.

## Training

### Windows

```
.\train_cmd.bat
```

### Linux
```
python train.py \
  --model_dir ./pretrained/pt_biobert1.1 \
  --train_dictionary_path ./datasets/ncbi-disease/train_dictionary.txt \
  --train_dir ./datasets/ncbi-disease/processed_traindev \
  --output_dir ./tmp/biosyn-ncbi-disease_sent \
  --topk 20 --epoch 10 --train_batch_size 16  \
  --initial_sparse_weight 0.2 --initial_sent_weight 0.1 \
  --learning_rate 1e-5 --max_length 25 --dense_ratio 0.5
```

### Pretrained

Pretrained model can be loaded from the `tmp/biosyn-ncbi-disease_sent` directory.

## TODO:

1. Allow model fine-tuning while training
2. Support special tokens to separate the entity from the sentence itself
