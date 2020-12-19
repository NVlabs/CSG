# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

import argparse
import os
import sys
import logging
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from pdb import set_trace as bp
from data.gta5 import GTA5
from data.cityscapes import Cityscapes
from model import csg_builder
from model.deeplab import ResNet as deeplab
from dataloader_seg import get_train_loader
from eval_seg import SegEvaluator
from utils.utils import get_params, IterNums, save_checkpoint, AverageMeter, lr_poly, adjust_learning_rate
from utils.logger import prepare_logger, prepare_seed
from utils.sgd import SGD

torch.backends.cudnn.enabled = True
CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
KLDivLoss = nn.KLDivLoss(reduction='batchmean')
best_mIoU = 0

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=6, type=int, dest='batch_size', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--csg', default=75., type=float, dest='csg', help="weight of LWF los (default: 0). Format: type('_')=>stage(',')")
parser.add_argument('--switch-model', default='deeplab50', choices=["deeplab50", "deeplab101"], help='which model to use')
parser.add_argument('--factor', default=0.1, type=float, dest='factor', help='scale factor of backbone learning rate (default: 0.1)')
parser.add_argument('--csg-stages', dest='csg_stages', default='4', help='resnet stages to involve in LWF, 0~4, seperated by dot')
parser.add_argument('--chunks', dest='chunks', default='8', help='stage-wise chunk to feature maps, seperated by dot')
parser.add_argument('--no-mlp', dest='mlp', action='store_false', default=True, help='not to use mlp during contrastive learning')
parser.add_argument('--apool', default=False, action='store_true', help='use A-Pool')
parser.add_argument('--augment', action='store_true', default=False, help='use augmentation')
parser.add_argument('--resume', default='none', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=19, type=int, dest='num_classes', help='the number of classes')
parser.add_argument('--gpus', default=0, type=int, help='gpu to use')
parser.add_argument('--evaluate', action='store_true', help='whether to use learn without forgetting (default: False)')
parser.add_argument('--save_dir', type=str, default="./runs", help='root folder to save checkpoints and log.')
parser.add_argument('--rand_seed', default=0, type=int, help='the number of classes')
parser.add_argument('--csg-k', default=65536, type=int, help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--timestamp', type=str, default='none', help='timestamp for logging naming')
parser.set_defaults(bottleneck=True)

best_mIoU = 0


def main():
    global args, best_mIoU
    PID = os.getpid()
    args = parser.parse_args()
    prepare_seed(args.rand_seed)
    device = torch.device("cuda:"+str(args.gpus))

    if args.timestamp == 'none':
        args.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    switch_model = args.switch_model
    assert switch_model in ["deeplab50", "deeplab101"]

    # Log outputs
    if args.evaluate:
        args.save_dir = args.save_dir + "/GTA5-%s-evaluate"%switch_model + \
            "%s/%s"%('/'+args.resume if args.resume != 'none' else '', args.timestamp)
    else:
        args.save_dir = args.save_dir + \
            "/GTA5_512x512-{model}-LWF.stg{csg_stages}.w{csg_weight}-APool.{apool}-Aug.{augment}-chunk{chunks}-mlp{mlp}.K{csg_k}-LR{lr}.bone{factor}-epoch{epochs}-batch{batch_size}-seed{seed}".format(
                    model=switch_model,
                    csg_stages=args.csg_stages,
                    mlp=args.mlp,
                    csg_weight=args.csg,
                    apool=args.apool,
                    augment=args.augment,
                    chunks=args.chunks,
                    csg_k=args.csg_k,
                    lr="%.2E"%args.lr,
                    factor="%.1f"%args.factor,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=args.rand_seed
                    ) + \
            "%s/%s"%('/'+args.resume if args.resume != 'none' else '', args.timestamp)
    logger = prepare_logger(args)

    from config_seg import config as data_setting
    data_setting.batch_size = args.batch_size
    train_loader = get_train_loader(data_setting, GTA5, test=False, augment=args.augment)

    args.stages = [int(stage) for stage in args.csg_stages.split('.')] if len(args.csg_stages) > 0 else []
    chunks = [int(chunk) for chunk in args.chunks.split('.')] if len(args.chunks) > 0 else []
    assert len(chunks) == 1 or len(chunks) == len(args.stages)
    if len(chunks) < len(args.stages):
        chunks = [chunks[0]] * len(args.stages)

    if switch_model == 'deeplab50':
        layers = [3, 4, 6, 3]
    elif switch_model == 'deeplab101':
        layers = [3, 4, 23, 3]
    model = csg_builder.CSG(deeplab, get_head=None, K=args.csg_k, stages=args.stages, chunks=chunks, task='new-seg',
                              apool=args.apool, mlp=args.mlp,
                              base_encoder_kwargs={'num_seg_classes': args.num_classes, 'layers': layers})

    threds = 3
    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), args.num_classes, np.array([0.485, 0.456, 0.406]),
                    np.array([0.229, 0.224, 0.225]), model.encoder_q, [1, ], False, devices=args.gpus, config=data_setting, threds=threds,
                    verbose=False, save_path=None, show_image=False)  # just calculate mIoU, no prediction file is generated
                    # verbose=False, save_path="./prediction_files", show_image=True, show_prediction=True)  # generate prediction files


    # Setup optimizer
    factor = args.factor
    sgd_in = [
        {'params': get_params(model.encoder_q, ["conv1"]), 'lr': factor*args.lr},
        {'params': get_params(model.encoder_q, ["bn1"]), 'lr': factor*args.lr},
        {'params': get_params(model.encoder_q, ["layer1"]), 'lr': factor*args.lr},
        {'params': get_params(model.encoder_q, ["layer2"]), 'lr': factor*args.lr},
        {'params': get_params(model.encoder_q, ["layer3"]), 'lr': factor*args.lr},
        {'params': get_params(model.encoder_q, ["layer4"]), 'lr': factor*args.lr},
        {'params': get_params(model.encoder_q, ["fc_new"]), 'lr': args.lr},
        ]
    base_lrs = [ group['lr'] for group in sgd_in ]
    optimizer = SGD(sgd_in, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume != 'none':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            msg = model.load_state_dict(checkpoint['state_dict'])
            print("resume weights: ", msg)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=ImageClassdata> no checkpoint found at '{}'".format(args.resume))

    model = model.to(device)

    if args.evaluate:
        mIoU = validate(evaluator, model, -1)
        print(mIoU)
        exit(0)

    # Main training loop
    iter_max = args.epochs * len(train_loader)
    iter_stat = IterNums(iter_max)
    for epoch in range(args.start_epoch, args.epochs):
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, args.save_dir))
        logger.log("Epoch: %d"%(epoch+1))
        # train for one epoch
        train(args, train_loader, model, optimizer, base_lrs, iter_stat, epoch, logger, device, adjust_lr=epoch<args.epochs)

        # evaluate on validation set
        torch.cuda.empty_cache()
        mIoU = validate(evaluator, model, epoch)
        logger.writer.add_scalar("mIoU", mIoU, epoch+1)
        logger.log("mIoU: %f"%mIoU)

        # remember best mIoU and save checkpoint
        is_best = mIoU > best_mIoU
        best_mIoU = max(mIoU, best_mIoU)
        save_checkpoint(args.save_dir, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mIoU': best_mIoU,
        }, is_best)

    logging.info('Best accuracy: {mIoU:.3f}'.format(mIoU=best_mIoU))


def train(args, train_loader, model, optimizer, base_lrs, iter_stat, epoch, logger, device, adjust_lr=True):
    tb_interval = 50

    csg_weight = args.csg

    """Train for one epoch on the training set"""
    losses = AverageMeter()
    losses_csg = [AverageMeter() for _ in range(len(model.stages))]  # [_loss] x #stages
    top1_csg = [AverageMeter() for _ in range(len(model.stages))]

    model.eval()
    model.encoder_q.fc_new.train()

    # train for one epoch
    optimizer.zero_grad()
    epoch_size = len(train_loader)
    train_loader_iter = iter(train_loader)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(epoch_size), file=sys.stdout, bar_format=bar_format, ncols=80)

    lr = lr_poly(base_lrs[-1], iter_stat.iter_curr, iter_stat.iter_max, 0.9)
    logger.log("lr %f"%lr)
    for idx_iter in pbar:

        optimizer.zero_grad()
        if adjust_lr:
            lr = lr_poly(base_lrs[-1], iter_stat.iter_curr, iter_stat.iter_max, 0.9)
            adjust_learning_rate(base_lrs, optimizer, iter_stat.iter_curr, iter_stat.iter_max, 0.9)

        sample = next(train_loader_iter)
        label = sample['label'].to(device)
        input = sample['data']
        if args.augment:
            input_q = input.to(device)
            input_k = sample['img_k'].to(device)
        else:
            input_q = input.to(device)
            input_k = None

        # keys: output, predictions_csg, targets_csg
        results = model(input_q, input_k)

        # synthetic task
        loss = CrossEntropyLoss(results['output'], label.long())
        # measure accuracy and record loss
        losses.update(loss, label.size(0))
        for idx in range(len(model.stages)):
            _loss = 0
            acc1 = None
            # predictions: cosine b/w q and k
            # targets: zeros
            _loss = CrossEntropyLoss(results['predictions_csg'][idx], results['targets_csg'][idx])
            acc1, acc5 = accuracy_ranking(results['predictions_csg'][idx].data, results['targets_csg'][idx], topk=(1, 5))
            loss = loss + _loss * csg_weight
            if acc1 is not None: top1_csg[idx].update(acc1, label.size(0))
            # measure accuracy and record loss
            losses_csg[idx].update(_loss, label.size(0))

        loss.backward()

        # compute gradient and do SGD step
        optimizer.step()
        # increment iter number
        iter_stat.update()

        if idx_iter % tb_interval == 0: logger.writer.add_scalar("loss/ce", losses.val, idx_iter + epoch * epoch_size)
        description = "[XE %.3f]"%(losses.val)
        description += "[CSG "
        loss_str = ""
        acc_str = ""
        for idx, stage in enumerate(model.stages):
            if idx_iter % tb_interval == 0: logger.writer.add_scalar("loss/layer%d"%stage, losses_csg[idx].val, idx_iter + epoch * epoch_size)
            loss_str += "%.2f|"%losses_csg[idx].val
            if idx_iter % tb_interval == 0: logger.writer.add_scalar("prec/layer%d"%stage, top1_csg[idx].val[0], idx_iter + epoch * epoch_size)
            acc_str += "%.1f|"%top1_csg[idx].val[0]
        description += "loss:%s ranking:%s]"%(loss_str[:-1], acc_str[:-1])
        if idx_iter % tb_interval == 0: logger.writer.add_scalar("loss/total", losses.val + sum([_loss.val for _loss in losses_csg]), idx_iter + epoch * epoch_size)
        pbar.set_description("[Step %d/%d][%s]"%(idx_iter + 1, epoch_size, str(csg_weight)) + description)


def validate(evaluator, model, epoch):
    with torch.no_grad():
        model.eval()
        # _, mIoU = evaluator.run_online()
        _, mIoU = evaluator.run_online_multiprocess()
    return mIoU


def accuracy_ranking(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
