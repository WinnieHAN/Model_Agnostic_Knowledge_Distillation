from __future__ import print_function

import sys
import os
import time
import argparse
import uuid
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
from seq2seq_rl.seq2seq import Seq2seq_Model
from seq2seq_rl.rl import LossRL, DistLossGECRL, get_bleu, get_correct
import pickle, random

import spacy
import torch
from torchtext import data, datasets
from word_level.bridge_of_weqi_v import weiqi_predict

import fairseq_cli.train
sys.path.append(".")
sys.path.append("..")

def main(args, checkpoint_name, wf_src_name, wf_trg_name):
    spacy_en = spacy.load('en_core_web_sm')  # python -m spacy download en

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    def tokenizer(text):  # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    src_field = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=150, include_lengths=True, batch_first=True)  #use_vocab=False
    trg_field = src_field
    seq2seq_train_data = datasets.TranslationDataset(
        path=os.path.join('data', 'debpe', 'sample.src-trg'), exts=('.src', '.trg'),
        fields=(src_field, trg_field))
    seq2seq_dev_data = datasets.TranslationDataset(
        path=os.path.join('data', 'debpe', 'sample.src-trg'), exts=('.src', '.trg'),
        fields=(src_field, trg_field))
    src_field.build_vocab(seq2seq_train_data, max_size=80000)  # ,vectors="glove.6B.100d"
    # trg_field.build_vocab(seq2seq_train_data, max_size=80000)
    # mt_dev shares the fields, so it shares their vocab objects

    train_iter = data.BucketIterator(
        dataset=seq2seq_train_data, batch_size=64,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=-1, shuffle=False)  # Note that if you are runing on CPU, you must set device to be -1, otherwise you can leave it to 0 for GPU.
    # dev_iter = data.BucketIterator(
    #     dataset=seq2seq_dev_data, batch_size=64,
    #     sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=-1, shuffle=False)
    num_words = src_field.vocab.itos.__len__()  # ?? word_embedd ??
    word_dim = 300  #??
    hidden_size = 256
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=None, device=device).to(device)  # TODO: random init vocab
    # seq2seq.emb.weight.requires_grad = False
    print(seq2seq)

    # loss_seq2seq = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    # parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    # optim_seq2seq = torch.optim.Adam(parameters_need_update, lr=0.0002)

    seq2seq.load_state_dict(torch.load(checkpoint_name))  # TODO: 10.7
    seq2seq.to(device)
    itos = src_field.vocab.itos
    wf_src = open(wf_src_name, 'w', encoding="utf-8")
    wf_trg = open(wf_trg_name, 'w', encoding="utf-8")
    for _, batch in enumerate(train_iter):  # for _ in range(1, num_batches + 1):  word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_dev, batch_size, unk_replace=unk_replace)  # word:(32,50)  char:(32,50,35)
        word, lenghts_src = batch.src
        trg, lenghts_trg = batch.trg
        inp, _ = seq2seq.add_noise(word, lenghts_src)
        sel, _ = seq2seq(inp.long().to(device), LEN=word.size()[1])
        sel = sel.detach().cpu().numpy()
        lenghts_src = lenghts_src.data.numpy()
        for i in range(len(lenghts_src)):
            len_s = lenghts_src[i]
            a = [sel[i][j] for j in range(len_s)]
            line = ' '.join([itos[j] for j in a]) + '\n'
            wf_src.write(line)

            a = [word[i][j] for j in range(len_s)]
            line = ' '.join([itos[j] for j in a]) + '\n'
            wf_src.write(line)

            len_t = lenghts_trg[i]
            a = [trg[i][j] for j in range(len_t)]
            line = ' '.join([itos[j] for j in a]) + '\n'
            wf_trg.write(line)
            wf_trg.write(line)

    wf_src.close()
    wf_trg.close()

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--cuda', action='store_true', help='using GPU')
    args = args_parser.parse_args()
    seq2seq_load_path = 'checkpoint_v/model'
    checkpoint_name = seq2seq_load_path + str(0) + '.pt'
    wfs = 'generated.src-trg.src' #os.path.join('')
    wft = 'generated.src-trg.trg' #os.path.join('')
    main(args, checkpoint_name, wf_src_name=wfs, wf_trg_name=wft)