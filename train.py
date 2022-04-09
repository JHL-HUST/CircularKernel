import os
import sys
import time
import glob
import torch
import utils
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model import NetworkCIFAR as Network
from utils import *



parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./dataset/', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save_epoch_freq', type=int, default=100, help='num of training epochs')
parser.add_argument('--AUTO_RESUME', action='store_true', help='auto resume after the program stops')
parser.add_argument('--RESUME', type=str, default='', help='the file to resume the program')
parser.add_argument('--start_epoch', type=int, default=0, help='the starting epoch count')
parser.add_argument('--ROTATE', action='store_true', help='to rotate the validation set')
parser.add_argument('--degree', type=int, default=0, help='the maximum degree to rotate the image')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
args = parser.parse_args()


if args.save=='checkpoints/':
  args.save = '{}eval-{}'.format(args.save, args.arch)

utils.create_exp_dir(args.save)


CIFAR_CLASSES = 10

logger = create_logger(output_dir=args.save)


if args.set=='cifar100':
    CIFAR_CLASSES = 100

def main():

  if not torch.cuda.is_available():
    logger.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logger.info('gpu device = %d' % args.gpu)
  logger.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)

  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  ###############


  model_without_ddp = model
  logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar(args)

  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)


  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

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
    max_accuracy = load_checkpoints(args, model_without_ddp, optimizer, scheduler, logger)


  start=time.time()
  for epoch in range(args.start_epoch, args.epochs):

    logger.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logger.info('train_acc %f', train_acc)

    if epoch % 10 == 0 or epoch == args.epochs - 1:
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logger.info('valid_acc %f', valid_acc)
    scheduler.step() 

    if epoch % args.save_epoch_freq == 0 or epoch == (
            args.epochs - 1):  # cache our model every <save_epoch_freq> epochs
      print('saving the model at the end of epoch %d, iters %d' % (epoch, args.epochs))

      save_checkpoints(args, epoch, model_without_ddp, optimizer, scheduler, logger)
    for parameter_group in optimizer.param_groups:
      lr = parameter_group['lr']
      logger.info(f'learning rates = {lr:.7f}')
    duration_epoch = time.time() - start
    logger.info('Total searching time: %ds', duration_epoch)
  duration = time.time() - start
  logger.info('Total searching time: %ds', duration)



def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():    
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda(non_blocking=True)
      logits,_ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

