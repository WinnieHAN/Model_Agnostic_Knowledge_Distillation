#!/bin/bash

. /home/projects/11001764/wenjuan/gec_wj/beast19/transformer/common/base.sh

dev_file=$data_wq/test/wilocABCN-dev/wilocABCN-dev
output_file=$1
# ${dev_file}.tok.src 
testset=wilocABCN-dev
# $transformer/common/eval.sh ${dev_file}.tok.src  ${dev_file}.m2
#/home/projects/11001764/wenjuan/gec_wj/2top_matched.dev.1best
#/home/projects/11001764/wenjuan/gec_wj/beast19/data_wq/test/wilocABCN-dev/wilocABCN-dev.tok.src

$transformer/common/eval_dis.sh $output_file $data_wq/test/$testset/$testset.m2
