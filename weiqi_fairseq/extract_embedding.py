import collections
import itertools
import os, errno
import math
import torch
import subprocess
import argparse

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils
from fairseq.fp16_trainer import FP16Trainer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from torch.serialization import default_restore_location

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', help='model-path used for extracting word embedding')
parser.add_argument('--dict-path', help='dict-path used for extracting word embedding')
parser.add_argument('--save-path', help='save-path used for saving word embedding')
args = parser.parse_args()

def load_checkpoint():
	filename = args.model_path
	if os.path.isfile(filename):
		state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
		dict_size = len(state['model']['decoder.embed_tokens.weight'])
		word_dim = state['model']['decoder.embed_tokens.weight'].size()[1]
		# print(state['model']['decoder.embed_tokens.weight'].tolist()[:10])
		dict = []
		with open(args.dict_path, encoding='utf-8') as f:
			for line in f:
				dict.append(' '.join(line.strip().split(' ')[:-1]))
		assert len(dict) == dict_size, \
			"The size of dict(--dict-path) must be equal to the size of Tensor(['model']['decoder.embed_tokens.weight']) "
		with open(args.save_path, 'w', encoding='utf-8') as f1:
			f1.write(str(dict_size)+' '+str(word_dim)+'\n')
			for i in range(dict_size):
				cache = [str(float(item)) for item in state['model']['decoder.embed_tokens.weight'][i]]
				# print(dict[i])
				embedding = ' '.join(cache)
				f1.write(dict[i]+' '+embedding+'\n')

load_checkpoint()
