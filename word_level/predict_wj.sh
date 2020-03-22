#!/bin/bash
. ${HOME}/.bashrc
#. /home/project/11001764/gec_wj/beast19/transformer/common/base.sh
#export LANG="C.UTF-8"

project=/home/projects/11001764
scratch=$HOME/scratch
# change the currPath
gec=$project/gec_wj
acl_scripts=$gec/neural-naacl2018/training/scripts
scripts=$gec/beast19/scripts
software=$gec/beast19/software
fairseqpy=$software/fairseq-transformer
m2scorer=$software/m2scorer
errant=$software/errant
moses=$software/mosesdecoder
nbest_reranker=$software/nbest-reranker-errant
transformer=$gec/beast19/transformer
data_wq=$gec/beast19/data_wq


output=${outputdir}/${testset}.out
logfile=log.test.${testset}.txt
tmp=$scratch/gec/$$
beam=12
nbest=$beam
threads=12


models=/home/projects/11001764/gec_sy/beast19/transformer/fine_tune/kakao+newscrawl-synt-50mlines+SSE.maxtokens3000.drop1/epoch5/seed0/st19-train-corrected/maxtokens4500/checkpoint7.pt
dictdir=$data_wq/st19-train-corrected/processed


conda activate gec
CUDA_VISIBLE_DEVICES=0 python $fairseqpy/interactive.py --no-progress-bar  --path $models --beam $beam --nbest $nbest --replace-unk --num-shards $threads $dictdir < /home/project/11001764/gec_wj/beast19/transformer/common/wilocABCN-test-1.tok.processed.bpe.src | tail -n +6 > /home/project/11001764/gec_wj/beast19/transformer/common/tmp.tmp.txt




cat /home/project/11001764/gec_wj/beast19/transformer/common/tmp.tmp.txt | grep "^H"  | python3 -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if i%${nbest} == 0 ]); print(x)" | cut -f3 > /home/project/11001764/gec_wj/beast19/transformer/common/output.bpe.txt

conda activate py2.7.17
cat /home/project/11001764/gec_wj/beast19/transformer/common/output.bpe.txt | sed 's|@@ ||g' | sed '$ d' | bash /home/projects/11001764/gec_wj/neural-naacl2018/training/scripts/postprocess_safe.sh > /home/project/11001764/gec_wj/beast19/transformer/common/output.word.txt
