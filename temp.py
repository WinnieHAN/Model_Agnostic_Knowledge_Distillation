import os

# a = open('dev.reference_bea').read().split('\n')
# wf = open('dev.reference', 'w')
# b = []
# for idx in range(len(a)):
#     b.append(a[idx])
#     b.append(a[idx])
#
# wf.write('\n'.join(b))
#
# wf.close()

# wf = open('dev.reference', 'w')
# wf.write('[SEP]')
# for idx in range(2046714-1):
#     wf.write('\n')
#     wf.write('[SEP]')
# 
# 
# wf.close()


os.system('rm -rf matched.nbest matched.1best matched output.1best.scores output.1best.bleu_out output.1best.ter_out output.1best.ter_out.sum output.1best.meteor_out output.1best.args output.nbest output.1best decoder_config zmert dev.matched matched_dev.nbest matched_dev.1best matched_dev')
