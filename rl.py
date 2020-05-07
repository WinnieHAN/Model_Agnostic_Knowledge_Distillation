from nltk.translate.bleu_score import sentence_bleu as BLEU
import numpy as np
import torch.nn as nn
import torch, os, codecs, math
# ref = [[1, 2, 3, 4, 5, 6]]
# cnd = [1, 3, 4, 5, 6]
# bleu = BLEU(ref, cnd)
#
# print('BLEU: %.4f%%' % (bleu * 100))


def get_bleu(out, dec_out, vocab_size):
    out = out.tolist()
    dec_out = dec_out.tolist()
    stop_token = 1
    if stop_token in out:
        cnd = out[:out.index(stop_token)]
    else:
        cnd = out

    if stop_token in dec_out:
        ref = [dec_out[:dec_out.index(stop_token)]]
    else:
        ref = [dec_out]

    bleu = BLEU(ref, cnd)

    return bleu


def get_correct(out, dec_out, num_words):
    out = out.tolist()
    dec_out = dec_out.tolist()
    stop_token = 1
    if stop_token in out:
        cnd = out[:out.index(stop_token)]
    else:
        cnd = out

    if stop_token in dec_out:
        ref = [dec_out[:dec_out.index(stop_token)]]
    else:
        ref = [dec_out]
    tmp = [1 if cnd[i] == ref[i] else 0 for i in range(1, min(len(cnd), len(ref)))]
    if not tmp:
        stc_crt = 0
    else:
        stc_crt = sum(tmp)
    if not max(len(cnd), len(ref)) - 1>0:
        print(max(len(cnd), len(ref)))
    # assert max(len(cnd), len(ref)) - 1>0
    return stc_crt, max(len(cnd), len(ref))-1



class departureLossRL(nn.Module): # parsers
    def __init__(self, device, word_alphabet, vocab_size):
        super(departureLossRL, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet
        self.vocab_size = vocab_size


    def get_reward_departure(self, golden_out, stc_length_out, departure_out_list):
        stc_dda = sum([0 if golden_out[i] == departure_out_list[i] else 1 for i in range(0, stc_length_out)])

        reward = stc_dda

        return reward


    def write_text(self, ori_words, ori_words_length, sel, stc_length_out):
        condsf = 'cands.txt'
        refs = 'refs.txt'
        oris = [[self.word_alphabet.get_instance(ori_words[si, wi]).encode('utf-8') for wi in range(1, ori_words_length[si])] for si in range(len(ori_words))]
        preds = [[self.word_alphabet.get_instance(sel[si, wi]).encode('utf-8') for wi in range(1, stc_length_out[si])] for si in range(len(sel))]

        wf = codecs.open(condsf, 'w', encoding='utf8')
        preds_tmp = [' '.join(i) for i in preds]
        preds_s = '\n'.join(preds_tmp)
        wf.write(preds_s)
        wf.close()

        wf = codecs.open(refs, 'w', encoding='utf8')
        oris_tmp = [' '.join(i) for i in oris]
        oris_s = '\n'.join(oris_tmp)
        wf.write(oris_s)
        wf.close()


    def forward(self, sel, pb, golden_out, mask_id, stc_length_out, departure_out_list):
        ####1####
        batch = sel.shape[0]
        rewards_z1 = []
        for i in range(batch):  #batch
            reward = self.get_reward_departure(golden_out, stc_length_out[i], departure_out_list)  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
        rewards_z1 = np.asarray(rewards_z1)

        ####3#####add meaning_preservation as reward
        # batch = sel.shape[0]
        # self.write_text(ori_words, ori_words_length, sel, stc_length_out)
        # os.system('/home/hanwj/anaconda3/envs/bertscore/bin/python seq2seq_rl/get_bertscore_ppl.py')
        # logppl = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp_ppl.txt') # * (-0.1)
        # ppl = -np.exp(logppl) * 0.001


        #-----------------------------------------------

        rewards = (rewards_z1)*0.001      #TODO  ppl +
        # rewards = bleus_w * 10  # 8.26

        ls3 = 0
        cnt3 = 0
        stc_length_seq = sel.shape[1]
        for j in range(stc_length_seq):
            wgt3 = np.asarray([1 if j < min(stc_length_out[i]+1, stc_length_seq) else 0 for i in range(batch)])  # consider in STOP token
            ls3 += (- pb[:, j] *
                    torch.from_numpy(rewards-self.bl).float().to(self.device) *  # rewards-self.bl
                    torch.from_numpy(wgt3.astype(float)).float().to(self.device)).sum()
            cnt3 += np.sum(wgt3)

        ls3 /= cnt3
        rewards_ave3 = np.average(rewards)
        self.bl = (self.bl * self.bn + rewards_ave3) / (self.bn + 1)
        self.bn += 1


        loss = ls3

        return loss, np.average(rewards_z1), None #np.average(ppl) #loss, ls, ls1, bleu, bleu1