import sys
import os, math, importlib, itertools, nltk
from nltk.translate.bleu_score import SmoothingFunction

# reload is a buildin in python2. use importlib.reload in python 3

# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')   # Try setting the system default encoding as utf-8 at the start of the script, so that all strings are encoded using that. Or there will be UnicodeDecodeError: 'ascii' codec can't decode byte...

sys.path.append(".")
sys.path.append("..")

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
#from word_level.bridge_of_weqi_v import weqi_predict


def test_v_model(hidden_size, train_iter, dev_iter, device, num_words, seq2seq_load_path):
    word_dim = 300  #
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=None,
                            device=device).to(device)  # TODO: random init vocab
    seq2seq.load_state_dict(torch.load(os.path.join(seq2seq_load_path, 'model'+ str(80) + '.pt')))  # TODO: 10.7
    seq2seq.to(device)

    if True:  # i%1 == 0:
        seq2seq.eval()
        bleu_ep = 0
        acc_numerator_ep = 0
        acc_denominator_ep = 0
        testi = 0
        for _, batch in enumerate(dev_iter):  # for _ in range(1, num_batches + 1):  word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_dev, batch_size, unk_replace=unk_replace)  # word:(32,50)  char:(32,50,35)
            src, lengths_src = batch.src
            trg, lengths_trg = batch.trg
            inp = src
            # inp, _ = seq2seq.add_noise(word, lengths)
            dec_out = trg  # TODO: hanwj
            sel, _ = seq2seq(inp, LEN=inp.size()[1]+5)
            sel = sel.detach().cpu().numpy()
            dec_out = dec_out.cpu().numpy()

            bleus = []
            for j in range(sel.shape[0]):
                bleu = get_bleu(sel[j], dec_out[j], EOS_IDX)  # sel
                bleus.append(bleu)
                numerator, denominator = get_correct(sel[j], dec_out[j], EOS_IDX)
                acc_numerator_ep += numerator
                acc_denominator_ep += denominator  # .detach().cpu().numpy() TODO: 10.8
            bleu_bh = np.average(bleus)
            bleu_ep += bleu_bh
            testi += 1
        bleu_ep /= testi  # num_batches
        print('testi: ', testi)
        print('Valid bleu: %.4f%%' % (bleu_ep * 100))
        # print(acc_denominator_ep)
        print('Valid acc: %.4f%%' % ((acc_numerator_ep * 1.0 / acc_denominator_ep) * 100))


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

if __name__ == '__main__':
    spacy_en = spacy.load('en_core_web_sm')  # python -m spacy download en
    src_field = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True, eos_token='<eos>')  # , fix_length=150 use_vocab=False   fix_length=20,
    trg_field = src_field
    seq2seq_train_data = datasets.TranslationDataset(
        path=os.path.join('data', 'debpe', 'sample.src-trg'), exts=('.src', '.trg'),
        fields=(src_field, trg_field))
    print('training stcs loaded')
    seq2seq_dev_data = datasets.TranslationDataset(
        path=os.path.join('data', 'debpe', 'valid.src-trg'), exts=('.src', '.trg'),
        fields=(src_field, trg_field))
    src_field.build_vocab(seq2seq_train_data, max_size=80000)  # ,vectors="glove.6B.100d"

    # trg_field.build_vocab(seq2seq_train_data, max_size=80000)
    # mt_dev shares the fields, so it shares their vocab objects
    device = torch.device('cuda')

    train_iter = data.BucketIterator(
        dataset=seq2seq_train_data, batch_size=64,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=device, shuffle=True)  # Note that if you are runing on CPU, you must set device to be -1, otherwise you can leave it to 0 for GPU.
    dev_iter = data.BucketIterator(
        dataset=seq2seq_dev_data, batch_size=64,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=device, shuffle=False)
    hidden_size = 256
    num_words = len(src_field.vocab.stoi)
    PAD_IDX = src_field.vocab.stoi['<pad>']
    EOS_IDX = src_field.vocab.stoi['<eos>']
    seq2seq_save_path = 'checkpoint_v'
    test_v_model(hidden_size, train_iter, dev_iter, device, num_words, seq2seq_save_path)