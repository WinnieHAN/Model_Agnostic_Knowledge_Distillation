from __future__ import print_function

__author__ = 'max'
"""
Implementation
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
import torchtext
from word_level.bridge_of_weqi_v import weiqi_predict, weiqi_predict_rerank
# import torchtext.data.Fields
# import fairseq_cli.train
import tqdm

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
    args_parser.add_argument('--v_model_load_path', default='checkpoint_generator/model', type=str,
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
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    def tokenizer(text):  # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # src_field = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True, eos_token='<eos>')  #use_vocab=False
    src_field = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True, eos_token='<eos>')  #use_vocab=False
    trg_field = src_field
    # from torchtext.datasets.
    seq2seq_train_data = datasets.TranslationDataset(
        path=os.path.join('data', 'debpe', 'train.src-trg'), exts=('.src', '.trg'),
        fields=(src_field, trg_field))
    seq2seq_dev_data = datasets.TranslationDataset(
        path=os.path.join('data', 'debpe', 'valid.src-trg'), exts=('.src', '.trg'),
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
    PAD_IDX = src_field.vocab.stoi['<pad>']
    EOS_IDX = src_field.vocab.stoi['<eos>']
    UNK_IDX = src_field.vocab.stoi['<unk>']
    word_dim = 300  # ??
    v_model = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=None,
                            device=device).to(device)
    # v_model.load_state_dict(torch.load(args.v_model_load_path + str(0) + '.pt'))  # TODO: 7.13--20

    # v_model.load_state_dict(torch.load('checkpoint_generator/seq2seq_save_model0.pt'))  # TODO: 7.13--20
    v_model.to(device)

    # Pretrain seq2seq model using denoising autoencoder. model name: seq2seq model
    print('Pretrain seq2seq model.')
    EPOCHS = 0  # 150
    DECAY = 0.97
    num_words = len(src_field.vocab.stoi)  # ?? word_embedd ??
    word_dim = 300  # ??
    seq2seq = Seq2seq_Model(EMB=word_dim, HID=args.hidden_size, DPr=0.5, vocab_size=num_words, word_embedd=None, device=device).to(device)  # TODO: random init vocab
    print(seq2seq) # seq2seq.emb.weight.requires_grad = False
    loss_seq2seq = torch.nn.CrossEntropyLoss(reduction='none').to(device)     # criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    parameters_need_update = filter(lambda p: p.requires_grad, seq2seq.parameters())
    optim_seq2seq = torch.optim.Adam(parameters_need_update, lr=0.0002)
    # seq2seq.load_state_dict(torch.load(args.seq2seq_load_path + str(0) + '.pt'))  # TODO: 10.7--0
    seq2seq.to(device)

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
        print('----------------epoch '+str(epoch_i)+'---------------------')
        ls_rl_ep = rewards1 = ls_re_seq2seq_ep = 0
        for batch_i, batch in enumerate(train_iter):
            print('--------train_iter %s--------'%(str(batch_i)))
            v_model.eval()
            src, lengths_src = batch.trg  # word:(32,50)  150,64
            trg, lengths_trg = batch.src
            batch_size = src.size()[0]
            max_len = src.size()[1]  # batch_first
            masks = torch.arange(max_len).expand(len(lengths_src), max_len) < lengths_src.unsqueeze(1)
            masks = masks.long().to(device)
            if True:  # inp.size()[1]<15:#True:  #inp.size()[1]<15:
                _, sel, pb = seq2seq(src.long().to(device), is_tr=True, M=M, LEN=5+src.size()[1])
                sel1 = sel.data.detach()
                try:
                    end_position = torch.eq(sel1, EOS_IDX).nonzero()  # TODO: hanwj END_token=0
                except RuntimeError:
                    continue
                lengths_sel, _ = length_masks(sel1, batch_size, end_position)
                out_pred, _ = v_model(sel, LEN=sel.size()[1]+5)
                sample_wf = 'weiqi_f/sample.tok.src'
                idx_to_words(sel, EOS_IDX, PAD_IDX, src_field.vocab.itos, sample_wf)
                sudo_golden_out_words = weiqi_predict_rerank(sample_wf)  #
                sudo_golden_out_words = [s.strip('\n')+' <eos>' for s in sudo_golden_out_words]
                sudo_golden_out = [[src_field.vocab.stoi[w] for w in s.split()] for s in sudo_golden_out_words]  #not batch, list
                ls_rl_bh, reward1, _ = loss_gec_rl(sel, pb, predicted_out=out_pred, stc_length_out=lengths_sel, sudo_golden_out=sudo_golden_out, EOS_IDX=EOS_IDX)  # sudo_heads_pred_1 TODO: (sel, pb, heads)  # heads is replaced by dec_out.long().to(device)
                optim_bia_rl.zero_grad()
                ls_rl_bh.backward()
                optim_bia_rl.step()
                ls_rl_bh = ls_rl_bh.cpu().detach().numpy()
                ls_rl_ep += ls_rl_bh
                rewards1 += reward1

            retrain_v_model = True
            if retrain_v_model:
                trg1 = list_to_tensors(sudo_golden_out, PAD_IDX)
                v_model.train()

                dec_inp = torch.cat((torch.ones(size=[batch_size, 1]).long().to(device), trg1[:, 0:-1].to(device)),dim=1)  # 1  = pad_idx
                out = v_model.forward(sel1.to(device), is_tr=True, dec_inp=dec_inp.long().to(device))
                out = out.view((out.shape[0] * out.shape[1], out.shape[2]))
                trg1 = trg1.view((trg1.shape[0] * trg1.shape[1],))
                ls_seq2seq_bh = loss_seq2seq(out, trg1.long().to(device))  # 9600, 8133
                ls_seq2seq_bh = ls_seq2seq_bh.sum() / ls_seq2seq_bh.numel()

                trg2 = src
                # trg2_leng = lengths_src
                dec_inp2 = torch.cat((torch.ones(size=[batch_size, 1]).long().to(device), trg2[:, 0:-1].to(device)),dim=1)  # 1  = pad_idx
                out2 = v_model.forward(sel1.to(device), is_tr=True, dec_inp=dec_inp2.long().to(device))
                out2 = out2.view((out2.shape[0] * out2.shape[1], out2.shape[2]))
                trg2 = trg2.view((trg2.shape[0] * trg2.shape[1],))
                ls_seq2seq_bh2 = loss_seq2seq(out2, trg2.long().to(device))  # 9600, 8133
                ls_seq2seq_bh2 = ls_seq2seq_bh2.sum() / ls_seq2seq_bh2.numel()

                loss = ls_seq2seq_bh + ls_seq2seq_bh2
                optim_seq2seq.zero_grad()
                loss.backward()
                optim_seq2seq.step()
                loss = loss.cpu().detach().numpy()
                ls_re_seq2seq_ep += loss
                # v_model.forward(src=sel1, src_leng=lengths_sel, trg1=trg1, trg1_leng=trg1_leng, trg2=src, trg2_leng=lengths_src)  # trg1 e_model_out  trg2 gold
                if batch_i//5000 == 0:
                    torch.save(seq2seq.state_dict(), args.seq2seq_save_path + '_epoch_' + str(epoch_i) + '_batch_' + str(batch_i) + '.pt')
                    torch.save(v_model.state_dict(), args.v_model_save_path + '_epoch_' + str(epoch_i) + '_batch_' + str(batch_i) + '.pt')
        print('test loss: ', ls_rl_ep)
        print('test reward parser b: ', rewards1)
        print('ls_re_seq2seq_ep: ', ls_re_seq2seq_ep)


def list_to_tensors(sudo_golden_out, PAD_IDX):
    sudo_golden_out[0] = [4,4]
    a = [torch.tensor(ii) for ii in sudo_golden_out]
    b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True, padding_value=PAD_IDX)
    # torch.nn.utils.rnn.pad_sequence(sudo_golden_out, batch_first=True, padding_value=-1)
    trg1, trg1_leng = None, None
    return b


def length_masks(sel1, batch_size, end_position):
    masks_sel = torch.ones_like(sel1, dtype=torch.float)
    lengths_sel = torch.ones(batch_size).fill_(sel1.shape[1])  # sel1.shape[1]-1 TODO: because of end token in the end
    if not len(end_position) == 0:
        ij_back = -1
        for ij in end_position:
            if not (ij[0] == ij_back):
                lengths_sel[ij[0]] = ij[1] + 1  # + 1
                masks_sel[ij[0], ij[1] + 1:] = 0  # +1 TODO: because of end token in the end
                ij_back = ij[0]
    return lengths_sel, masks_sel


def idx_to_words(sel, EOS_IDX, PAD_IDX, itos, sample_wf):
    sel = sel.tolist()

    # stop_token = PAD_IDX
    # for idx in range(len(sel)):
    #     if stop_token in sel[idx]:
    #         sel[idx] = sel[idx][:sel[idx].index(stop_token)]

    stop_token = EOS_IDX
    for idx in range(len(sel)):
        if stop_token in sel[idx]:
            sel[idx] = sel[idx][:sel[idx].index(stop_token)]

    cnd = [[itos[ij] for ij in ii] for ii in sel]

    f = open(sample_wf, 'wb')
    lines=[' '.join(i) for i in cnd]
    texts='\n'.join(lines)
    f.write(texts.encode('utf-8'))
    f.close()
    return cnd


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    main()
