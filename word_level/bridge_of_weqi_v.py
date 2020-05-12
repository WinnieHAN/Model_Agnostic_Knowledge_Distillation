import os, sys
from shutil import copyfile

def weiqi_predict(batch):
    os.system('bash predict_wj.sh')
    f = open('output.word.txt', 'r').read()
    return f

def weiqi_predict_rerank(batchf):
    if not os.path.exists('weiqi_f/sample.tok.src'):
        return False 
    elif not open('weiqi_f/sample.tok.src', 'rb').readlines():
        return False
    else:
        try:
            os.system('bash ' + os.path.join('word_level', 'predict_wj_rerank.sh'))  #  + ' &>/home/projects/11001764/wenjuan/gec_distillation/weiqi_f/weqi.log'
            # /home/projects/11001764/wenjuan/gec_distillation/weiqi_f/ensemble/reranking+bert+numpunct.wilocABCN-dev.errant/sample.out.txt
            copyfile('weiqi_f/ensemble/reranking+bert+numpunct.wilocABCN-dev.errant/sample.out.txt', 'weiqi_f/output.word.txt')
            words_list = open('weiqi_f/output.word.txt', 'rb').readlines()
            os.system('rm '+ batchf)
            os.system('rm '+ 'weiqi_f/output.word.txt')
            if len(words_list)<4:  # the least batch size
                return False
            return words_list
        except:
            print('ERROR!!!!!!!!!!')
            return False

if __name__ == '__main__':
    k = weiqi_predict_rerank(os.path.join('weiqi_f', 'sample.tok.src'))
    print(k)