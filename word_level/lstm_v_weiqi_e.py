from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os, math, codecs

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from word_level.bridge_of_weqi_v import weqi_predict

uid = uuid.uuid4().hex[:6]

#--------------------------------------
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import os
import random
import sys

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.trainer import Trainer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    while (
        lr > args.min_lr
        and epoch_itr.next_epoch_idx <= max_epoch
        and trainer.get_num_updates() < max_update
    ):
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        # early stop
        if should_stop_early(args, valid_losses[0]):
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            break

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        )
    train_meter.stop()
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))
    return model

def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs >= args.patience


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args.log_interval == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)

            # reset mid-epoch stats after each log interval
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters('train_inner')

        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    # train v model
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        network = main(args)

    # generate adv examples
    print('Load victim model.')
    # network.load_state_dict(torch.load('models/parsing/biaffine/network.pt'))  # TODO: 7.13
    # network.to(device)
    network.train()
    FLAG = True
    for para in network.parameters():
        if FLAG:
            para.requires_grad = True
            FLAG = False
        else:
            para.requires_grad = False

    outf = 'word_level/adv_sentences025.txt'
    wf = codecs.open(outf, 'w', encoding='utf8')
    outf_conllu = 'word_level/adv_sentences025.conllu'
    wf_conllu = codecs.open(outf_conllu, 'w', encoding='utf8')
    ori_outf = 'word_level/ori_sentences.txt'
    ori_wf = codecs.open(ori_outf, 'w', encoding='utf8')
    ori_word_embedding = network.parameters().next().detach().data.cpu().numpy().T  # torch.Size([35374, 100])
    #-------------------------------------------------------------------------------------------------------------------
        kk = 0
        print('batch_size: ', str(batch_size))
        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as model_scope:  # reuse=tf.AUTO_REUSE or reuse=True
                dataset = load_datasets(load_existing_dump=True)
                config = dataset.model_config
                model = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix,
                                    dataset.dep_embedding_matrix)
                saver = tf.train.Saver()
                model_dir = 'params_2020-01-28'
                ckpt_path = tf.train.latest_checkpoint(os.path.join(DataConfig.data_dir_path, model_dir))
                saver.restore(sess, ckpt_path)
                for batch in conllx_data.iterate_batch_tensor(data_train, batch_size):  # TODO: data_train
                    kk = kk + 1
                    print('--------'+str(kk)+'--------')
                    test_data = dataform_biaf2fastdep(batch)
                    seudo_heads = fastdep_predict(test_data=test_data, dataset=dataset, sess=sess, model=model, batch_max_length=batch[0].shape[1])

                    #----------------
                    optim.zero_grad()
                    seudo_heads = torch.from_numpy(np.array(seudo_heads)).long().to(device)
                    word, char, pos, heads, types, masks, lengths = batch
                    loss_arc, loss_type = network.loss(word, char, pos, seudo_heads, types, mask=masks, length=lengths)  # head should be replaced with seudo_heads
                    loss = loss_arc  #+  loss_type
                    loss = - loss
                    loss.backward()
                    clip_grad_norm_(network.parameters(), clip)
                    optim.step()

                    for batch_i in range(len(word)):
                        ori_wf.write(' '.join([str(word_alphabet.get_instance(wordi)) for wordi in word[batch_i][1:] if not wordi==1]).encode('utf-8'))
                        ori_wf.write('\n')
                    # adv_word_embedding = network.parameters().next().detach().data.cpu().numpy().T  #torch.Size([35374, 100])
                    for batch_i in range(len(word)):
                        changed_num = 0
                        for batch_j in range(1, lengths[batch_i].cpu().numpy()-1):
                            one_word = word[batch_i][batch_j].item()
                            if one_word==1:
                                continue
                            src_word_idx = one_word
                            adv_vector = network.parameters().next().detach().data.cpu().numpy()[src_word_idx,:]
                            ori_word_embedding[:, src_word_idx] = adv_vector
                            c_ = -2 * np.dot(ori_word_embedding.T, adv_vector)
                            a = np.sum(np.square(ori_word_embedding), axis=0)#.reshape((1, -1))
                            b = np.sum(np.square(adv_vector))
                            dist = a + c_ + b
                            neighbours, _ = glove_utils.pick_most_similar_words_from_vector(src_word_idx, adv_vector, dist, ret_count=3, threshold=20)
                            if word[batch_i][batch_j] == neighbours[0]:
                                continue
                            if changed_num*1.0/(lengths[batch_i].cpu().numpy()-1)>0.1:
                                break
                            word[batch_i][batch_j] = neighbours[0]
                            changed_num = changed_num + 1

                    for batch_i in range(len(word)):
                        wf.write(' '.join([str(word_alphabet.get_instance(wordi)) for wordi in word[batch_i][1:] if not wordi==1]))
                        wf.write('\n')
        wf.close()
        ori_wf.close()



if __name__ == '__main__':
    cli_main()
