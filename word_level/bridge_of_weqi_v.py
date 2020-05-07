import os, sys

def weiqi_predict(batch):
    os.system('bash predict_wj.sh')
    f = open('output.word.txt', 'r').read()

    return f


if __name__ == '__main__':
    weqi_predict()
