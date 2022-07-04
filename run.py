import os

from fairseq import interactive

os.system('python fairseq-transformer/interactive.py --no-progress-bar  --path checkpoints_e/checkpoint7.pt --beam 23 --nbest 12 --replace-unk --num-shards 23 checkpoints/processed < checkpoints/wilocABCN-test-1.tok.processed.bpe.src | tail -n +6 > tmp.tmp.txt')