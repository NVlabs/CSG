# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

import argparse
import os
import sys
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.visda17 import VisDA17
from data.loader_csg import TwoCropsTransform
from model.resnet import resnet101
from model import csg_builder
from utils.utils import get_params, IterNums, save_checkpoint, AverageMeter, lr_poly, adjust_learning_rate, accuracy
from utils.logger import prepare_logger, prepare_seed
from utils.sgd import SGD
from utils.augmentations import RandAugment, augment_list

torch.backends.cudnn.enabled = True
CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--data', default='/home/chenwy/taskcv-2017-public/classification/data', help='path to dataset')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, dest='batch_size', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--csg', default=0.1, type=float, dest='csg', help="weight of CSG loss (default: 0.1).")
parser.add_argument('--factor', default=0.1, type=float, dest='factor', help='scale factor of backbone learning rate (default: 0.1)')
parser.add_argument('--csg-stages', dest='csg_stages', default='4', help='resnet stages to involve in CSG, 0~4, seperated by dot')
parser.add_argument('--chunks', dest='chunks', default='1', help='stage-wise chunk to feature maps, seperated by dot')
parser.add_argument('--no-mlp', dest='mlp', action='store_false', default=True, help='not to use mlp during contrastive learning')
parser.add_argument('--apool', default=False, action='store_true', help='use A-Pool')
parser.add_argument('--augment', action='store_true', default=False, help='use augmentation')
parser.add_argument('--resume', default='none', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=12, type=int, dest='num_classes', help='the number of classes')
parser.add_argument('--evaluate', action='store_true', help='whether to use learn without forgetting (default: False)')
parser.add_argument('--save_dir', type=str, default="./runs", help='root folder to save checkpoints and log.')
parser.add_argument('--rand_seed', default=0, type=int, help='the number of classes')
parser.add_argument('--csg-k', default=65536, type=int, help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--timestamp', type=str, default='none', help='timestamp for logging naming')
parser.set_defaults(bottleneck=True)

best_prec1 = 0


def main():
    global args, best_prec1
    PID = os.getpid()
    args = parser.parse_args()
    prepare_seed(args.rand_seed)

    if args.timestamp == 'none':
        args.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    # Log outputs
    if args.evaluate:
        args.save_dir = args.save_dir + "/Visda17-Res101-evaluate" + \
            "%s/%s"%('/'+args.resume.replace('/', '+') if args.resume != 'none' else '', args.timestamp)
    else:
        args.save_dir = args.save_dir + \
            "/VisDA-Res101-CSG.stg{csg_stages}.w{csg_weight}-APool.{apool}-Aug.{augment}-chunk{chunks}-mlp{mlp}.K{csg_k}-LR{lr}.bone{factor}-epoch{epochs}-batch{batch_size}-seed{seed}".format(
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
            "%s/%s"%('/'+args.resume.replace('/', '+') if args.resume != 'none' else '', args.timestamp)
    logger = prepare_logger(args)

    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    if args.augment:
        data_transforms['train'] = transforms.Compose([
            RandAugment(1, 6., augment_list),
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        data_transforms['train'] = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    kwargs = {'num_workers': 20, 'pin_memory': True}
    if args.augment:
        # two source
        trainset = VisDA17(txt_file=os.path.join(args.data, "train/image_list.txt"), root_dir=os.path.join(args.data, "train"),
                           transform=TwoCropsTransform(data_transforms['train'], data_transforms['train']))
    else:
        # one source
        trainset = VisDA17(txt_file=os.path.join(args.data, "train/image_list.txt"), root_dir=os.path.join(args.data, "train"), transform=data_transforms['train'])
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    valset = VisDA17(txt_file=os.path.join(args.data, "validation/image_list.txt"), root_dir=os.path.join(args.data, "validation"), transform=data_transforms['val'], label_one_hot=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)

    args.stages = [int(stage) for stage in args.csg_stages.split('.')] if len(args.csg_stages) > 0 else []
    chunks = [int(chunk) for chunk in args.chunks.split('.')] if len(args.chunks) > 0 else []
    assert len(chunks) == 1 or len(chunks) == len(args.stages)
    if len(chunks) < len(args.stages):
        chunks = [chunks[0]] * len(args.stages)

    def get_head(num_ftrs, num_classes):
        _dim = 512
        return nn.Sequential(
           nn.Linear(num_ftrs, _dim),
           nn.ReLU(inplace=False),
           nn.Linear(_dim, num_classes),
        )
    model = csg_builder.CSG(
            resnet101, get_head=get_head, K=args.csg_k, stages=args.stages, chunks=chunks,
            apool=args.apool, mlp=args.mlp,
            )

    train_blocks = "conv1.bn1.layer1.layer2.layer3.layer4.fc"
    train_blocks = train_blocks.split('.')
    # Setup optimizer
    factor = args.factor
    sgd_in = []
    for name in train_blocks:
        if name != 'fc':
            sgd_in.append({'params': get_params(model.encoder_q, [name]), 'lr': factor*args.lr})
        else:
            # no update to fc but to fc_new
            sgd_in.append({'params': get_params(model.encoder_q, ["fc_new"]), 'lr': args.lr})
            if model.mlp:
                sgd_in.append({'params': get_params(model.encoder_q, ["fc_csg"]), 'lr': args.lr})
    base_lrs = [ group['lr'] for group in sgd_in ]
    optimizer = SGD(sgd_in, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume != 'none':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("resume weights: ", msg)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=ImageClassdata> no checkpoint found at '{}'".format(args.resume))

    model = model.cuda()

    if args.evaluate:
        prec1 = validate(val_loader, model, args, 0)
        print(prec1)
        exit(0)

    # Main training loop
    iter_max = args.epochs * len(train_loader)
    iter_stat = IterNums(iter_max)
    for epoch in range(args.start_epoch, args.epochs):
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, args.save_dir))
        logger.log("Epoch: %d"%(epoch+1))
        train(train_loader, model, optimizer, base_lrs, iter_stat, epoch, logger, args, adjust_lr=epoch<args.epochs)

        prec1 = validate(val_loader, model, args, epoch)
        logger.writer.add_scalar("prec", prec1, epoch+1)
        logger.log("prec: %f"%prec1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(args.save_dir, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, keep_last=1)

    logging.info('Best accuracy: {prec1:.3f}'.format(prec1=best_prec1))


def train(train_loader, model, optimizer, base_lrs, iter_stat, epoch, logger, args, adjust_lr=True):
    tb_interval = 50

    csg_weight = args.csg

    losses = AverageMeter()  # loss on target task
    losses_csg = [AverageMeter() for _ in range(len(model.stages))]  # [_loss] x #stages
    top1_csg = [AverageMeter() for _ in range(len(model.stages))]

    model.eval()

    # train for one epoch
    optimizer.zero_grad()
    epoch_size = len(train_loader)
    train_loader_iter = iter(train_loader)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(epoch_size), file=sys.stdout, bar_format=bar_format, ncols=80)

    lr = lr_poly(base_lrs[-1], iter_stat.iter_curr, iter_stat.iter_max, 0.9)
    logger.writer.add_scalar("lr", lr, epoch)
    logger.log("lr %f"%lr)
    for idx_iter in pbar:
        optimizer.zero_grad()
        if adjust_lr:
            lr = lr_poly(base_lrs[-1], iter_stat.iter_curr, iter_stat.iter_max, 0.9)
            adjust_learning_rate(base_lrs, optimizer, iter_stat.iter_curr, iter_stat.iter_max, 0.9)

        input, label = next(train_loader_iter)
        if args.augment:
            input_q = input[0].cuda()
            input_k = input[1].cuda()
        else:
            input_q = input.cuda()
            input_k = None
        label = label.cuda()

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
            # loss_csg[_type].append(_loss)
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


def validate(val_loader, model, args, epoch):
    """Perform validation on the validation set"""
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    val_size = len(val_loader)
    val_loader_iter = iter(val_loader)
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(val_size), file=sys.stdout, bar_format=bar_format, ncols=140)
    with torch.no_grad():
        for idx_iter in pbar:
            input, label = next(val_loader_iter)

            input = input.cuda()
            label = label.cuda()

            # compute output
            output, _ = model.encoder_q(input, task='new')
            output = torch.sigmoid(output)
            output = (output + torch.sigmoid(model.encoder_q(torch.flip(input, dims=(3,)), task='new')[0])) / 2

            # accumulate accuracyk
            prec1, gt_num = accuracy(output.data, label, args.num_classes, topk=(1,))
            top1.update(prec1[0], gt_num[0])

            description = "[Acc@1-mean: %.2f][Acc@1-cls: %s]"%(top1.vec2sca_avg, str(top1.avg.numpy().round(1)))
            pbar.set_description("[Step %d/%d]"%(idx_iter + 1, val_size) + description)

    logging.info(' * Prec@1 {top1.vec2sca_avg:.3f}'.format(top1=top1))
    logging.info(' * Prec@1 {top1.avg}'.format(top1=top1))

    return top1.vec2sca_avg


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
