import os, sys
from shutil import copyfile

def weiqi_predict(batch):
    os.system('bash predict_wj.sh')
    f = open('output.word.txt', 'r').read()
    return f

def weiqi_predict_rerank(batchf):
    # os.system('bash predict_wj.sh')
    # copyfile('weiqi_bf/ensemble/reranking+bert+numpunct.wilocABCN-dev.errant/sample.out.txt', 'weiqi_f/output.word.txt')
    words_list = open('weiqi_f/output.word.txt', 'r').readlines()
    return words_list

if __name__ == '__main__':
    weiqi_predict_rerank()
