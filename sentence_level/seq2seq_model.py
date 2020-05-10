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


def train_v_model(hidden_size, train_iter, dev_iter, device, num_words, seq2seq_save_path):
    EPOCHS = 100  # 150
    DECAY = 0.97
    word_dim = 300  #
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=None,
                            device=device).to(device)  # TODO: random init vocab
    loss_seq2seq = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_seq2seq = torch.optim.Adam(parameters_need_update, lr=0.0002)

    is_only_eval = False
    if is_only_eval:
        seq2seq.load_state_dict(torch.load('checkpoint_seq2seq/model10.pt'))

    seq2seq.to(device)
    for i in range(EPOCHS):
        ls_seq2seq_ep = 0
        seq2seq.train()
        # seq2seq.emb.weight.requires_grad = False
        print('----------' + str(i) + ' iter----------')
        for _, batch in enumerate(train_iter):
            src, lengths_src = batch.src  # word:(32,50)  150,64
            trg, lengths_trg = batch.trg
            batch_size = src.size()[0]
            # max_len = src.size()[1]  # batch_first
            # masks = torch.arange(max_len).expand(len(lengths_src), max_len) < lengths_src.unsqueeze(1)
            # masks = masks.long().to(device)
            # inp, _ = seq2seq.add_noise(src, lengths_src)
            inp = src
            dec_out = trg
            dec_inp = torch.cat((torch.ones(size=[batch_size,1]).long().to(device), trg[:, 0:-1]), dim=1)  # maybe wrong
            # train_seq2seq
            out = seq2seq(inp.long().to(device), is_tr=True, dec_inp=dec_inp.long().to(device))

            out = out.view((out.shape[0] * out.shape[1], out.shape[2]))
            dec_out = dec_out.view((dec_out.shape[0] * dec_out.shape[1],))
            # wgt = seq2seq.add_stop_token(masks, lengths_src)  # TODO
            # wgt = wgt.view((wgt.shape[0] * wgt.shape[1],)).float().to(device)
            # wgt = masks.view(-1)

            ls_seq2seq_bh = loss_seq2seq(out, dec_out.long().to(device))  # 9600, 8133
            ls_seq2seq_bh = ls_seq2seq_bh.sum() / ls_seq2seq_bh.numel()
            # ls_seq2seq_bh = (ls_seq2seq_bh * wgt).sum() / wgt.sum()

            optim_seq2seq.zero_grad()
            ls_seq2seq_bh.backward()
            optim_seq2seq.step()

            ls_seq2seq_bh = ls_seq2seq_bh.cpu().detach().numpy()
            ls_seq2seq_ep += ls_seq2seq_bh
        print('ls_seq2seq_ep: ', ls_seq2seq_ep)
        for pg in optim_seq2seq.param_groups:
            pg['lr'] *= DECAY

        # test th bleu of seq2seq
        if True:  # i%1 == 0:
            seq2seq.eval()
            bleu_ep = 0
            acc_numerator_ep = 0
            acc_denominator_ep = 0
            testi = 0
            for _, batch in enumerate(dev_iter):  # for _ in range(1, num_batches + 1):  word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_tensor(data_dev, batch_size, unk_replace=unk_replace)  # word:(32,50)  char:(32,50,35)
                src, lengths_src = batch.src
                trg, lengths_trg = batch.trg
                # inp, _ = seq2seq.add_noise(src, lengths_src)
                inp = src
                dec_out = trg
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
        # for debug TODO:
        if i%1 == 0:
            torch.save(seq2seq.state_dict(), os.path.join(seq2seq_save_path, 'model'+ str(i) + '.pt'))

    return seq2seq

def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

if __name__ == '__main__':
    spacy_en = spacy.load('en_core_web_sm')  # python -m spacy download en
    src_field = data.Field(sequential=True, tokenize=tokenizer, lower=False, include_lengths=True, batch_first=True, eos_token='<eos>')  # , fix_length=150 use_vocab=False   fix_length=20, init_token='<int>',
    trg_field = src_field
    seq2seq_train_data = datasets.TranslationDataset(
        path=os.path.join('data', 'debpe', 'train.src-trg'), exts=('.trg', '.src'),
        fields=(src_field, trg_field))
    print('training stcs loaded')
    seq2seq_dev_data = datasets.TranslationDataset(
        path=os.path.join('data', 'debpe', 'valid.src-trg'), exts=('.trg', '.src'),
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
    seq2seq_save_path = 'checkpoint_seq2seq'
    train_v_model(hidden_size, train_iter, dev_iter, device, num_words, seq2seq_save_path)