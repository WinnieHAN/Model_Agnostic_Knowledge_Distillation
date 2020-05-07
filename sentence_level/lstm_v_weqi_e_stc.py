from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

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
from word_level.bridge_of_weqi_v import weiqi_predict

import fairseq_cli.train

uid = uuid.uuid4().hex[:6]


# 3 sub-models should be pretrained in our approach
#   seq2seq pretrain, denoising autoencoder  | or using token-wise adv to generate adv examples.
#   structure prediction model
#   oracle parser
# then we train the seq2seq model using rl


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--cuda', action='store_true', help='using GPU')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    args_parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm')
    args_parser.add_argument('--objective', choices=['cross_entropy', 'crf'], default='cross_entropy',
                             help='objective function of training procedure.')
    args_parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--p_rnn', nargs=2, type=float, default=0.1, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0.,
                             help='The rate to replace a singleton word with UNK')
    #args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    # args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args_parser.add_argument('--v_model_save_path', default='checkpoint_v/model', type=str,
                             help='v_model_save_path')
    args_parser.add_argument('--v_model_load_path', default='checkpoint_v/model', type=str,
                             help='v_model_load_path')
    args_parser.add_argument('--seq2seq_save_path', default='checkpoint_generator/seq2seq_save_model', type=str,
                             help='seq2seq_save_path')
    args_parser.add_argument('--seq2seq_load_path', default='checkpoint_generator/seq2seq_save_model', type=str,
                             help='seq2seq_load_path')
    args_parser.add_argument('--rl_finetune_seq2seq_save_path', default='models/rl_finetune/seq2seq_save_model',
                             type=str, help='rl_finetune_seq2seq_save_path')
    args_parser.add_argument('--rl_finetune_network_save_path', default='models/rl_finetune/network_save_model',
                             type=str, help='rl_finetune_network_save_path')
    args_parser.add_argument('--rl_finetune_seq2seq_load_path', default='models/rl_finetune/seq2seq_save_model',
                             type=str, help='rl_finetune_seq2seq_load_path')
    args_parser.add_argument('--rl_finetune_network_load_path', default='models/rl_finetune/network_save_model',
                             type=str, help='rl_finetune_network_load_path')


    args_parser.add_argument('--direct_eval', action='store_true', help='direct eval without generation process')
    args = args_parser.parse_args()

    spacy_en = spacy.load('en_core_web_sm')  # python -m spacy download en

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

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
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=-1, shuffle=True)  # Note that if you are runing on CPU, you must set device to be -1, otherwise you can leave it to 0 for GPU.
    dev_iter = data.BucketIterator(
        dataset=seq2seq_dev_data, batch_size=64,
        sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)), device=-1, shuffle=False)

    # Train v model using ori examples. model name: vmodel
    print('Train v model using ori examples.')
    num_words = len(src_field.vocab.stoi)  # ?? word_embedd ??
    word_dim = 300  # ??
    v_model = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=None,
                            device=device).to(device)
    v_model.load_state_dict(torch.load(args.v_model_load_path + str(20) + '.pt'))  # TODO: 7.13
    v_model.to(device)

    # Pretrain seq2seq model using denoising autoencoder. model name: seq2seq model
    print('Pretrain seq2seq model using denoising autoencoder.')
    EPOCHS = 0  # 150
    DECAY = 0.97
    num_words = len(src_field.vocab.stoi)  # ?? word_embedd ??
    word_dim = 300  # ??
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=None, device=device).to(device)  # TODO: random init vocab
    # seq2seq.emb.weight.requires_grad = False
    print(seq2seq)

    loss_seq2seq = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_seq2seq = torch.optim.Adam(parameters_need_update, lr=0.0002)

    seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(0) + '.pt'))  # TODO: 10.7
    seq2seq.to(device)

    def count_parameters(model: torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(seq2seq):,} trainable parameters')
    # PAD_IDX = TRG.vocab.stoi['<pad>']
    # criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    for i in range(EPOCHS):
        ls_seq2seq_ep = 0
        seq2seq.train()
        seq2seq.emb.weight.requires_grad = False
        print('----------' + str(i) + ' iter----------')
        for _, batch in enumerate(train_iter):
            word, lengths_src = batch.src  # word:(32,50)  150,64
            trg, lengths_trg = batch.trg

            max_len = word.size()[1]  # batch_first
            masks = torch.arange(max_len).expand(len(lengths_src), max_len) < lengths_src.unsqueeze(1)
            masks = masks.long()
            inp, _ = seq2seq.add_noise(word, lengths_src)
            dec_out = word
            dec_inp = torch.cat((word[:, 0:1], word[:, 0:-1]), dim=1)  # maybe wrong
            # train_seq2seq
            out = seq2seq(inp.long().to(device), is_tr=True, dec_inp=dec_inp.long().to(device))

            out = out.view((out.shape[0] * out.shape[1], out.shape[2]))
            dec_out = dec_out.view((dec_out.shape[0] * dec_out.shape[1],))
            # wgt = seq2seq.add_stop_token(masks, lengths_src)  # TODO
            # wgt = wgt.view((wgt.shape[0] * wgt.shape[1],)).float().to(device)
            wgt = masks.view(-1)

            ls_seq2seq_bh = loss_seq2seq(out, dec_out.long().to(device))  # 9600, 8133
            ls_seq2seq_bh = (ls_seq2seq_bh * wgt).sum() / wgt.sum()

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
                word, lenghts_src = batch.src
                inp, _ = seq2seq.add_noise(word, lengths_src)
                dec_out = word
                sel, _ = seq2seq(inp.long().to(device), LEN=word.size()[1])
                sel = sel.detach().cpu().numpy()
                dec_out = dec_out.cpu().numpy()

                bleus = []
                for j in range(sel.shape[0]):
                    bleu = get_bleu(sel[j], dec_out[j], num_words)  # sel
                    bleus.append(bleu)
                    numerator, denominator = get_correct(sel[j], dec_out[j], num_words)
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
        if i >= 0:
            torch.save(seq2seq.state_dict(), args.seq2seq_save_path + str(i) + '.pt')

    # Train seq2seq model using rl with reward of biaffine. model name: seq2seq model
    print('Train seq2seq model using rl with reward.')
    EPOCHS = 1  # 0  # 80
    DECAY = 0.97
    M = 1  # this is the size of beam searching in rl
    seq2seq.emb.weight.requires_grad = False
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_bia_rl = torch.optim.Adam(parameters_need_update, lr=1e-5)  # 1e-5 0.00005
    loss_gec_rl = DistLossGECRL(device=device, word_alphabet=src_field.vocab.stoi, vocab_size=num_words).to(device)
    seq2seq.train()

    # seq2seq.load_state_dict(torch.load(args.rl_finetune_seq2seq_load_path + str(4) + '.pt'))  # TODO: 7.13
    # seq2seq.to(device)
    # network.load_state_dict(torch.load(args.rl_finetune_network_load_path + str(4) + '.pt'))  # TODO: 7.13
    # network.to(device)

    for epoch_i in range(EPOCHS):
        for _, batch in enumerate(train_iter):
            word, lengths_src = batch.src  # word:(32,50)  64, 150
            trg, lengths_trg = batch.trg

            max_len = word.size()[1]  # batch_first
            masks = torch.arange(max_len).expand(len(lengths_src), max_len) < lengths_src.unsqueeze(1)
            masks = masks.long()
            if True:  # inp.size()[1]<15:#True:  #inp.size()[1]<15:
                _, sel, pb = seq2seq(word.long().to(device), is_tr=True, M=M, LEN=word.size()[1])
                sel1 = sel.data.detach()
                try:
                    end_position = torch.eq(sel1, 0).nonzero()  # TODO: hanwj END_token=0
                except RuntimeError:
                    continue
                masks_sel = torch.ones_like(sel1, dtype=torch.float)
                lengths_sel = torch.ones(sel1.size()[0]).fill_(sel1.shape[1])  # sel1.shape[1]-1 TODO: because of end token in the end
                if not len(end_position) == 0:
                    ij_back = -1
                    for ij in end_position:
                        if not (ij[0] == ij_back):
                            lengths_sel[ij[0]] = ij[1]
                            masks_sel[ij[0], ij[1]:] = 0  # -1 TODO: because of end token in the end
                            ij_back = ij[0]
                out_pred, _ = v_model(sel, LEN=sel.size()[1])
                sudo_golden_out =out_pred # weiqi_predict(sel1)  #
                ls_rl_bh, reward1, reward5 = loss_gec_rl(sel, pb, predicted_out=out_pred, stc_length_out=lengths_sel, sudo_golden_out=sudo_golden_out)  # sudo_heads_pred_1 TODO: (sel, pb, heads)  # heads is replaced by dec_out.long().to(device)
                optim_bia_rl.zero_grad()
                ls_rl_bh.backward()
                optim_bia_rl.step()
                ls_rl_bh = ls_rl_bh.cpu().detach().numpy()
                print(ls_rl_bh)


if __name__ == '__main__':
    main()
