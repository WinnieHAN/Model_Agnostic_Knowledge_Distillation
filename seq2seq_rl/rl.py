from nltk.translate.bleu_score import sentence_bleu as BLEU
import numpy as np
import torch.nn as nn
import torch, os, codecs, math
# ref = [[1, 2, 3, 4, 5, 6]]
# cnd = [1, 3, 4, 5, 6]
# bleu = BLEU(ref, cnd)
#
# print('BLEU: %.4f%%' % (bleu * 100))


def get_bleu(out, dec_out, EOS_IDX):
    if not isinstance(out, list):
        out = out.tolist()
    if not isinstance(dec_out, list):
        dec_out = dec_out.tolist()
    stop_token = EOS_IDX
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


def get_correct(out, dec_out, EOS_IDX):
    out = out.tolist()
    dec_out = dec_out.tolist()
    stop_token = EOS_IDX
    if stop_token in out:
        cnd = out[:out.index(stop_token)]
    else:
        cnd = out

    if stop_token in dec_out:
        ref = dec_out[:dec_out.index(stop_token)]
    else:
        ref = dec_out
    tmp = [1 if cnd[i] == ref[i] else 0 for i in range(0, min(len(cnd), len(ref)))]
    if not tmp:
        stc_crt = 0
    else:
        stc_crt = sum(tmp)
    if not max(len(cnd), len(ref)) - 1>0:
        pass
        # print(max(len(cnd), len(ref)))
    # assert max(len(cnd), len(ref)) - 1>0
    return stc_crt, max(len(cnd), len(ref))


class LossRL(nn.Module):
    def __init__(self):
        super(LossRL, self).__init__()

        self.bl = 0
        self.bn = 0

    def forward(self, sel, pb, dec_out, stc_length, vocab_size):
        ls = 0
        cnt = 0

        sel = sel.detach().cpu().numpy()
        dec_out = dec_out.cpu().numpy()

        batch = sel.shape[0]
        bleus = []
        for i in range(batch):
            bleu = get_bleu(sel[i], dec_out[i], vocab_size)

            bleus.append(bleu)
        bleus = np.asarray(bleus)

        wgt = np.asarray([1 for _ in range(batch)])
        for j in range(stc_length):
            ls += (- pb[:, j] *
                   torch.from_numpy(bleus - self.bl).float().cuda() *
                   torch.from_numpy(wgt.astype(float)).float().cuda()).sum()
            cnt += np.sum(wgt)
            stop_token = 1
            wgt = wgt.__and__(sel[:, j] != stop_token)  # vocab_size + 1

        ls /= cnt

        bleu = np.average(bleus)
        self.bl = (self.bl * self.bn + bleu) / (self.bn + 1)
        self.bn += 1

        return ls


class LossBiafRL(nn.Module):
    def __init__(self, device, word_alphabet, vocab_size):
        super(LossBiafRL, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet
        self.vocab_size = vocab_size

    def get_reward_diff(self, out, dec_out, length_out, ori_words, ori_words_length):
        stc_dda = sum([0 if out[i] == dec_out[i] else 1 for i in range(1, length_out)])

        reward = stc_dda

        return reward

    def get_reward_same(self, out, dec_out, length_out, ori_words, ori_words_length):
        stc_dda = sum([1 if out[i] == dec_out[i] else 0 for i in range(1, length_out)])

        reward = stc_dda

        return reward

    def get_same_bc(self, out, dec_out, dec_out_1, length_out, ori_words, ori_words_length):
        stc_dda = sum([1 if out[i] == dec_out[i] == dec_out_1[i] else 0 for i in range(1, length_out)])

        reward = stc_dda

        return reward

    def get_diff_bc(self, dec_out, dec_out_1, out, length_out, ori_words, ori_words_length):
        stc_dda = sum([0 if out[i] == dec_out[i] == dec_out_1[i] else 1 for i in range(1, length_out)])

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


    def forward(self, sel, pb, predicted_out, golden_out, mask_id, stc_length_out, sudo_golden_out, sudo_golden_out_1, ori_words, ori_words_length):
        ####1####
        batch = sel.shape[0]
        rewards_z1 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
        rewards_z1 = np.asarray(rewards_z1)

        #####2####
        batch = sel.shape[0]
        rewards_z2 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z2.append(reward)
        rewards_z2 = np.asarray(rewards_z2)

        #####3####
        batch = sel.shape[0]
        rewards_z3 = []
        for i in range(batch):  #batch
            reward = self.get_reward_same(sudo_golden_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z3.append(reward)
        rewards_z3 = np.asarray(rewards_z3)

        ####3#####add meaning_preservation as reward
        batch = sel.shape[0]
        self.write_text(ori_words, ori_words_length, sel, stc_length_out)
        os.system('/home/hanwj/anaconda3/envs/bertscore/bin/python seq2seq_rl/get_bertscore_ppl.py')
        meaning_preservation = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp.txt')*100
        logppl = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp_ppl.txt') # * (-0.1)
        ppl = -np.exp(logppl) * 0.001
        # rewards = meaning_preservation * 10  # affect more

        bleus_w = []
        for i in range(batch):
            bleu = get_bleu(ori_words[i], sel[i], self.vocab_size)

            bleus_w.append(bleu)
        bleus_w = np.asarray(bleus_w)

        #-----------------------------------------------

        rewards = (meaning_preservation + ppl + rewards_z1 + rewards_z2 + rewards_z3)*0.001      #TODO  0.1# + bleus_w*5
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

        # print('rewards_z1: ', np.average(rewards_z1))
        # print('rewards_z2: ', np.average(rewards_z2))
        # print('meaning_preservation: ', np.average(meaning_preservation))
        # print('ppl: ', np.average(ppl))

        # loss = ls1
        return loss, np.average(rewards_z1), np.average(rewards_z2), np.average(rewards_z3), np.average(meaning_preservation), np.average(ppl) #loss, ls, ls1, bleu, bleu1

    def forward_verbose(self, sel, pb, predicted_out, golden_out, mask_id, stc_length_out, sudo_golden_out, sudo_golden_out_1, ori_words, ori_words_length):
        ####1####
        batch = sel.shape[0]
        rewards_z1 = []
        metrics1 = []
        metricsall1 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
            metrics1.append(1.0-(reward*1.0/(stc_length_out[i].cpu().numpy()-1)))
            allsame = 1 if reward==0 else 0
            metricsall1.append(allsame)
        rewards_z1 = np.asarray(rewards_z1)
        metrics1 = np.asarray(metrics1)
        metricsall1 = np.asarray(metricsall1)

        #####2####
        batch = sel.shape[0]
        rewards_z2 = []
        metrics2 = []
        metricsall2 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z2.append(reward)
            metrics2.append(1.0-(reward*1.0/(stc_length_out[i].cpu().numpy()-1)))
            allsame = 1 if reward==0 else 0
            metricsall2.append(allsame)
        rewards_z2 = np.asarray(rewards_z2)
        metrics2 = np.asarray(metrics2)
        metricsall2 = np.asarray(metricsall2)

        #####3####
        batch = sel.shape[0]
        rewards_z3 = []
        metrics3 = []
        metricsall3 = []
        for i in range(batch):  #batch
            reward = self.get_reward_same(sudo_golden_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            metric3 = self.get_same_bc(sudo_golden_out[i], sudo_golden_out_1[i], predicted_out[i], stc_length_out[i],
                                       ori_words[i],
                                       ori_words_length[
                                           i])  # we now only consider a simple case. the result of a third-party parser should be added here.

            stc_diff = self.get_diff_bc(sudo_golden_out[i], sudo_golden_out_1[i], predicted_out[i], stc_length_out[i],
                                        ori_words[i],
                                        ori_words_length[
                                            i])  # we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z3.append(reward)
            metrics3.append(metric3*1.0/(stc_length_out[i].cpu().numpy()-1))
            allsame = 1 if stc_diff==0 else 0
            metricsall3.append(allsame)
        rewards_z3 = np.asarray(rewards_z3)
        metrics3 = np.array(metrics3)
        metricsall3 = np.asarray(metricsall3)

        ####3#####add meaning_preservation as reward
        batch = sel.shape[0]
        self.write_text(ori_words, ori_words_length, sel, stc_length_out)
        os.system('/home/hanwj/anaconda3/envs/bertscore/bin/python seq2seq_rl/get_bertscore_ppl.py')
        meaning_preservation = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp.txt')*100
        logppl = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp_ppl.txt') # * (-0.1)
        ppl = -np.exp(logppl) * 0.001
        metrics4 = meaning_preservation
        metrics5 = logppl

        # rewards = meaning_preservation * 10  # affect more

        bleus_w = []
        for i in range(batch):
            bleu = get_bleu(ori_words[i], sel[i], self.vocab_size)

            bleus_w.append(bleu)
        bleus_w = np.asarray(bleus_w)

        #-----------------------------------------------

        rewards = (meaning_preservation + ppl + rewards_z1 + rewards_z2 + rewards_z3)*0.001      #TODO  0.1# + bleus_w*5
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

        # print('rewards_z1: ', np.average(rewards_z1))
        # print('rewards_z2: ', np.average(rewards_z2))
        # print('meaning_preservation: ', np.average(meaning_preservation))
        # print('ppl: ', np.average(ppl))

        # loss = ls1
        return loss, np.average(rewards_z1), np.average(rewards_z2), np.average(rewards_z3), np.average(meaning_preservation), np.average(ppl) , np.sum(metrics1), np.sum(metrics2), np.sum(metrics3), np.sum(metrics4), np.sum(metrics5), np.sum(metricsall1), np.sum(metricsall2), np.sum(metricsall3)  # loss, ls, ls1, bleu, bleu1
 #loss, ls, ls1, bleu, bleu1



class TagLossBiafRL(nn.Module): # parsers
    def __init__(self, device, word_alphabet, vocab_size):
        super(TagLossBiafRL, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet
        self.vocab_size = vocab_size

    def get_reward_diff(self, out, dec_out, length_out, ori_words, ori_words_length):
        stc_dda = sum([0 if out[i] == dec_out[i] else 1 for i in range(0, length_out)])

        reward = stc_dda

        return reward

    def get_reward_same(self, out, dec_out, length_out, ori_words, ori_words_length):
        stc_dda = sum([1 if out[i] == dec_out[i] else 0 for i in range(0, length_out)])

        reward = stc_dda

        return reward

    def get_same_bc(self, out, dec_out, dec_out_1, length_out, ori_words, ori_words_length):
        stc_dda = sum([1 if out[i] == dec_out[i] == dec_out_1[i] else 0 for i in range(0, length_out)])

        reward = stc_dda

        return reward

    def get_diff_bc(self, dec_out, dec_out_1, out, length_out, ori_words, ori_words_length):
        stc_dda = sum([0 if out[i] == dec_out[i] == dec_out_1[i] else 1 for i in range(0, length_out)])

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


    def forward(self, sel, pb, predicted_out, golden_out, mask_id, stc_length_out, sudo_golden_out, sudo_golden_out_1, ori_words, ori_words_length):
        ####1####
        batch = sel.shape[0]
        rewards_z1 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
        rewards_z1 = np.asarray(rewards_z1)

        #####2####
        batch = sel.shape[0]
        rewards_z2 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z2.append(reward)
        rewards_z2 = np.asarray(rewards_z2)

        #####3####
        batch = sel.shape[0]
        rewards_z3 = []
        for i in range(batch):  #batch
            reward = self.get_reward_same(sudo_golden_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z3.append(reward)
        rewards_z3 = np.asarray(rewards_z3) *0.01

        ####3#####add meaning_preservation as reward
        batch = sel.shape[0]
        self.write_text(ori_words, ori_words_length, sel, stc_length_out)
        os.system('/home/hanwj/anaconda3/envs/bertscore/bin/python seq2seq_rl/get_bertscore_ppl.py')
        meaning_preservation = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp.txt')#*100
        logppl = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp_ppl.txt') # * (-0.1)
        ppl = -np.exp(logppl) * 0.001
        # rewards = meaning_preservation * 10  # affect more

        bleus_w = []
        for i in range(batch):
            bleu = get_bleu(ori_words[i], sel[i], self.vocab_size)

            bleus_w.append(bleu)
        bleus_w = np.asarray(bleus_w)

        #-----------------------------------------------

        rewards = (meaning_preservation + ppl + rewards_z1 + rewards_z2 + rewards_z3)*0.001      #TODO  0.1# + bleus_w*5
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

        # print('rewards_z1: ', np.average(rewards_z1))
        # print('rewards_z2: ', np.average(rewards_z2))
        # print('meaning_preservation: ', np.average(meaning_preservation))
        # print('ppl: ', np.average(ppl))

        # loss = ls1
        return loss, np.average(rewards_z1), np.average(rewards_z2), np.average(rewards_z3), np.average(meaning_preservation), np.average(ppl) #loss, ls, ls1, bleu, bleu1

    def forward_verbose(self, sel, pb, predicted_out, golden_out, mask_id, stc_length_out, sudo_golden_out, sudo_golden_out_1,
                ori_words, ori_words_length):
        ####1####tagging
        batch = sel.shape[0]
        rewards_z1 = []
        metrics1 = []
        metricsall1 = []
        for i in range(batch):  # batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out[i], stc_length_out[i], ori_words[i],
                                          ori_words_length[
                                              i])  # we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
            metrics1.append(1.0-(reward*1.0/(stc_length_out[i].cpu().numpy())))
            allsame = 1 if reward==0 else 0
            metricsall1.append(allsame)
        rewards_z1 = np.asarray(rewards_z1)
        metrics1 = np.asarray(metrics1)
        metricsall1 = np.asarray(metricsall1)


        #####2####
        batch = sel.shape[0]
        rewards_z2 = []
        metrics2 = []
        metricsall2 = []
        for i in range(batch):  # batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i],
                                          ori_words_length[
                                              i])  # we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z2.append(reward)
            metrics2.append(1.0-(reward*1.0/(stc_length_out[i].cpu().numpy())))
            allsame = 1 if reward==0 else 0
            metricsall2.append(allsame)
        rewards_z2 = np.asarray(rewards_z2)
        metrics2 = np.asarray(metrics2)
        metricsall2 = np.asarray(metricsall2)

        #####3####
        batch = sel.shape[0]
        rewards_z3 = []
        metrics3 = []
        metricsall3 = []
        for i in range(batch):  # batch
            reward = self.get_reward_same(sudo_golden_out[i], sudo_golden_out_1[i], stc_length_out[i], ori_words[i],
                                          ori_words_length[i])  # we now only consider a simple case. the result of a third-party parser should be added here.
            metric3 = self.get_same_bc(sudo_golden_out[i], sudo_golden_out_1[i], predicted_out[i], stc_length_out[i], ori_words[i],
                                          ori_words_length[i])  # we now only consider a simple case. the result of a third-party parser should be added here.

            stc_diff = self.get_diff_bc(sudo_golden_out[i], sudo_golden_out_1[i], predicted_out[i], stc_length_out[i], ori_words[i],
                                          ori_words_length[i])  # we now only consider a simple case. the result of a third-party parser should be added here.


            rewards_z3.append(reward)
            metrics3.append(metric3*1.0/(stc_length_out[i].cpu().numpy()))
            allsame = 1 if stc_diff==0 else 0
            metricsall3.append(allsame)
        rewards_z3 = np.asarray(rewards_z3) * 0.01
        metrics3 = np.array(metrics3)
        metricsall3 = np.asarray(metricsall3)

        ####3#####add meaning_preservation as reward
        batch = sel.shape[0]
        self.write_text(ori_words, ori_words_length, sel, stc_length_out)
        os.system('/home/hanwj/anaconda3/envs/bertscore/bin/python seq2seq_rl/get_bertscore_ppl.py')
        meaning_preservation = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp.txt')  # *100
        logppl = np.loadtxt('/home/hanwj/PycharmProjects/structure_adv/temp_ppl.txt')  # * (-0.1)
        ppl = -np.exp(logppl) * 0.001
        # rewards = meaning_preservation * 10  # affect more
        metrics4 = meaning_preservation
        metrics5 = logppl

        bleus_w = []
        for i in range(batch):
            bleu = get_bleu(ori_words[i], sel[i], self.vocab_size)

            bleus_w.append(bleu)
        bleus_w = np.asarray(bleus_w)

        # -----------------------------------------------

        rewards = (meaning_preservation + ppl + rewards_z1 + rewards_z2 + rewards_z3) * 0.001  # TODO  0.1# + bleus_w*5
        # rewards = bleus_w * 10  # 8.26

        ls3 = 0
        cnt3 = 0
        stc_length_seq = sel.shape[1]
        for j in range(stc_length_seq):
            wgt3 = np.asarray([1 if j < min(stc_length_out[i] + 1, stc_length_seq) else 0 for i in
                               range(batch)])  # consider in STOP token
            ls3 += (- pb[:, j] *
                    torch.from_numpy(rewards - self.bl).float().to(self.device) *  # rewards-self.bl
                    torch.from_numpy(wgt3.astype(float)).float().to(self.device)).sum()
            cnt3 += np.sum(wgt3)

        ls3 /= cnt3
        rewards_ave3 = np.average(rewards)
        self.bl = (self.bl * self.bn + rewards_ave3) / (self.bn + 1)
        self.bn += 1

        loss = ls3

        # print('rewards_z1: ', np.average(rewards_z1))
        # print('rewards_z2: ', np.average(rewards_z2))
        # print('meaning_preservation: ', np.average(meaning_preservation))
        # print('ppl: ', np.average(ppl))

        # loss = ls1
        return loss, np.average(rewards_z1), np.average(rewards_z2), np.average(rewards_z3), np.average(
            meaning_preservation), np.average(ppl), np.sum(metrics1), np.sum(metrics2), np.sum(metrics3), np.sum(metrics4), np.sum(metrics5), np.sum(metricsall1), np.sum(metricsall2), np.sum(metricsall3)  # loss, ls, ls1, bleu, bleu1



class DistLossBiafRL(nn.Module):
    def __init__(self, device, word_alphabet, vocab_size):
        super(DistLossBiafRL, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet
        self.vocab_size = vocab_size

    def get_reward_diff(self, out, dec_out, length_out, ori_words, ori_words_length):
        stc_dda = sum([0 if out[i] == dec_out[i] else 1 for i in range(1, length_out)])

        reward = stc_dda

        return reward

    def get_reward_same(self, out, dec_out, length_out, ori_words, ori_words_length):
        stc_dda = sum([1 if out[i] == dec_out[i] else 0 for i in range(1, length_out)])

        reward = stc_dda

        return reward

    def get_same_bc(self, out, dec_out, dec_out_1, length_out, ori_words, ori_words_length):
        stc_dda = sum([1 if out[i] == dec_out[i] == dec_out_1[i] else 0 for i in range(1, length_out)])

        reward = stc_dda

        return reward

    def get_diff_bc(self, dec_out, dec_out_1, out, length_out, ori_words, ori_words_length):
        stc_dda = sum([0 if out[i] == dec_out[i] == dec_out_1[i] else 1 for i in range(1, length_out)])

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


    def forward(self, sel, pb, predicted_out, golden_out, mask_id, stc_length_out, sudo_golden_out, sudo_golden_out_1, ori_words, ori_words_length):
        ####1####
        batch = sel.shape[0]
        rewards_z1 = []
        for i in range(batch):  #batch
            reward = self.get_reward_diff(predicted_out[i], sudo_golden_out[i], stc_length_out[i], ori_words[i], ori_words_length[i])  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
        rewards_z1 = np.asarray(rewards_z1)

        ####3#####add meaning_preservation as reward
        batch = sel.shape[0]
        self.write_text(ori_words, ori_words_length, sel, stc_length_out)
        os.system('/home/hanwj/anaconda3/envs/bertscore/bin/python seq2seq_rl/get_bertscore_ppl.py')
        logppl = np.loadtxt('temp_ppl.txt') # * (-0.1)
        ppl = -np.exp(logppl) * 0.001

        bleus_w = []
        for i in range(batch):
            bleu = get_bleu(ori_words[i], sel[i], self.vocab_size)

            bleus_w.append(bleu)
        bleus_w = np.asarray(bleus_w)

        #-----------------------------------------------

        rewards = (ppl + rewards_z1)*0.001      #TODO  0.1# + bleus_w*5

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

        # print('rewards_z1: ', np.average(rewards_z1))
        # print('rewards_z2: ', np.average(rewards_z2))
        # print('meaning_preservation: ', np.average(meaning_preservation))
        # print('ppl: ', np.average(ppl))

        # loss = ls1
        return loss, np.average(rewards_z1), np.average(ppl) #loss, ls, ls1, bleu, bleu1

class DistLossGECRL(nn.Module):
    def __init__(self, device, word_alphabet, vocab_size):
        super(DistLossGECRL, self).__init__()

        self.bl = 0
        self.bn = 0
        self.device = device
        self.word_alphabet = word_alphabet
        self.vocab_size = vocab_size

    def get_reward_departure(self, out, dec_out, EOS_IDX):
        # reward = get_bleu(out, dec_out, EOS_IDX)
        # stc_dda = sum([0 if out[i] == dec_out[i] else 1 for i in range(0, int(length_out.item()))])
        # reward = stc_dda
        # return reward

        if not isinstance(out, list):
            out = out.tolist()
        if not isinstance(dec_out, list):
            dec_out = dec_out.tolist()
        stop_token = EOS_IDX
        if stop_token in out:
            cnd = out[:out.index(stop_token)+1]  # consider in eos token in training
        else:
            cnd = out

        if stop_token in dec_out:
            ref = [dec_out[:dec_out.index(stop_token)+1]]  # consider in eos token in training
        else:
            ref = [dec_out]

        bleu = BLEU(ref, cnd)

        return 1 - bleu

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


    def forward(self, sel, pb, predicted_out, stc_length_out, sudo_golden_out, EOS_IDX):
        ####1####
        batch = sel.shape[0]
        rewards_z1 = []
        for i in range(batch):  #batch
            reward = self.get_reward_departure(predicted_out[i], sudo_golden_out[i], EOS_IDX)  #  we now only consider a simple case. the result of a third-party parser should be added here.
            rewards_z1.append(reward)
        rewards_z1 = np.asarray(rewards_z1)

        rewards = rewards_z1 #(rewards_z1)*0.001      #TODO  0.1# + bleus_w*5  (ppl + rewards_z1)*0.001

        ls3 = 0
        cnt3 = 0
        stc_length_seq = sel.shape[1]
        for j in range(stc_length_seq):
            wgt3 = np.asarray([1 if j < min(stc_length_out[i], stc_length_seq) else 0 for i in range(batch)])  # consider in STOP token  stc_length_out[i]+1
            ls3 += (- pb[:, j] *
                    torch.from_numpy(rewards-self.bl).float().to(self.device) *  # rewards-self.bl
                    torch.from_numpy(wgt3.astype(float)).float().to(self.device)).sum()
            cnt3 += np.sum(wgt3)

        ls3 /= cnt3
        rewards_ave3 = np.average(rewards)
        self.bl = (self.bl * self.bn + rewards_ave3) / (self.bn + 1)
        self.bn += 1

        loss = ls3

        # print('rewards_z1: ', np.average(rewards_z1))
        # print('rewards_z2: ', np.average(rewards_z2))
        # print('meaning_preservation: ', np.average(meaning_preservation))
        # print('ppl: ', np.average(ppl))

        # loss = ls1
        return loss, np.average(rewards_z1), None # np.average(ppl) #loss, ls, ls1, bleu, bleu1