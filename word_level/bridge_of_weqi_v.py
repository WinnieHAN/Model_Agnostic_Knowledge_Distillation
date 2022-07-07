import os, sys
from shutil import copyfile

def weiqi_predict(batch):
    os.system('bash predict_wj.sh')
    f = open('output.word.txt', 'r').read()
    return f

def weiqi_predict_rerank(batchf):
    teacher_model = 'gector'
    if teacher_model == 'weiqi':
        os.system('bash predict_wj.sh')
        copyfile('weiqi_bf/ensemble/reranking+bert+numpunct.wilocABCN-dev.errant/sample.out.txt', 'weiqi_f/output.word.txt')
        words_list = open('weiqi_f/output.word.txt', 'r').readlines()
    elif teacher_model == 'gector':
        copyfile('weiqi_f/sample.tok.src', 'teacher_model/gector/dump/input.txt')
        os.system('python teacher_model/gector/predict.py')
        copyfile('teacher_model/gector/dump/output.txt', 'weiqi_f/output.word.txt')
        words_list = open('weiqi_f/output.word.txt', 'r').readlines()
        # os.system('rm teacher_model/gector/dump/output.txt')
        os.system('rm teacher_model/gector/dump/input.txt')
    return words_list

if __name__ == '__main__':
    weiqi_predict_rerank()
