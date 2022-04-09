import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import NetworkImageNet as Network
from utils import *
from PIL import TiffImagePlugin


parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--data', type=str, default='imagenet', help='location of the data corpus')
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='gpu: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--tmp_data_dir', type=str, default='./dataset/imagenet', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--classes', type=int, default=1000, help='the classes of training dataset')
parser.add_argument('--save_epoch_freq', type=int, default=25, help='num of training epochs')
parser.add_argument('--AUTO_RESUME', action='store_true', help='auto resume after the program stops')
parser.add_argument('--RESUME', type=str, default='', help='the file to resume the program')
parser.add_argument('--start_epoch', type=int, default=0, help='the starting epoch count')

args, unparsed = parser.parse_known_args()


if args.save == 'checkpoints/':
    args.save = '{}eval-{}-{}-{}'.format(args.save, args.note, args.arch, 'imagenet')

print(args.save)
utils.create_exp_dir(args.save)

CLASSES = args.classes


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


logger = create_logger(output_dir=args.save)


def main():


    


    if not torch.cuda.is_available():
        logger.info('No GPU device available')
        sys.exit(1)

    gpu = args.gpu
    str_ids = gpu.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])


    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logger.info("args = %s", args)
    logger.info("unparsed_args = %s", unparsed)
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logger.info(genotype)
    print('--------------------------')


    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)

    num_gpus = len(gpu_ids)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if False:
            model = torch.nn.DataParallel(model, gpu_ids)
            model.to(gpu_ids[0])
        else:
            model.to(gpu_ids[0])
            model = torch.nn.DataParallel(model, gpu_ids)



    model_without_ddp = model  # .NetworkImageNet
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))



    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    data_dir = args.tmp_data_dir
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    if args.AUTO_RESUME:
        resume_file = auto_resume_helper(args.save)
        if resume_file:
            if args.RESUME:
                logger.warning(f"auto-resume changing resume file from {args.RESUME} to {resume_file}")

            args.RESUME = resume_file

            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {args.save}, ignoring auto resume')

    if args.RESUME:
        max_accuracy = load_checkpoints(args, model, optimizer, scheduler, logger)


    best_acc_top1 = 0
    best_acc_top5 = 0
    lr = args.learning_rate
    expriment_start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        logger.info('Epoch: %d lr %e', epoch, current_lr)
        print('Epoch:{} lr {:.7f}'.format(epoch, current_lr))
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logger.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        if num_gpus > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()

        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)

        logger.info('Train_acc: %f', train_acc)
        epoch_duration = time.time() - epoch_start
        logger.info('Epoch time: %ds.', epoch_duration)
        if epoch % args.save_epoch_freq == 0 or epoch == (args.epochs - 1):
            valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
            logger.info('Valid_acc_top1: %f', valid_acc_top1)
            logger.info('Valid_acc_top5: %f', valid_acc_top5)

            is_best = False
            if valid_acc_top5 > best_acc_top5:
                best_acc_top5 = valid_acc_top5
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save)


            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, args.epochs))
            save_checkpoints(args, epoch, model, optimizer, scheduler, logger)
    logger.info('End of expriment Time Taken: %d sec' % (time.time() - expriment_start_time))


def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)

        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))


        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)



        if step % args.report_freq == 0:

            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logger.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                         step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)
            #print('TRAIN Step: {} Objs: {:.4f} R1: {:.4f} R5: {:.4f} Duration: {:.4f}s BTime: {:.4f}s'.format(step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg))


    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logger.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg,
                         duration)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main() 
