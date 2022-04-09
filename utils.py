import os
import numpy as np
import torch
import torch.nn as nn
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
import logging
import functools
from auto_augment import CIFAR10Policy
import math

def save_checkpoints(config, epoch, model, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    # if config.AMP_OPT_LEVEL != "O0":
    #     save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.save, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def load_checkpoints(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.RESUME}....................")
    if config.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    #logger.info(msg)
    max_accuracy = 0.0
    #optimizer=optimizer[0]
    #lr_scheduler=lr_scheduler[0]
    # import code
    # code.interact(local=locals())
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:  #config.phase == 'train' and
        # import code
        # code.interact(local=locals())
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #config.defrost()
        config.start_epoch = checkpoint['epoch']+1
        #config.freeze()
        logger.info(f"=> loaded successfully '{config.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_checkpoints_test(config, model, logger):
    logger.info(f"==============> Resuming form {config.RESUME}....................")
    if config.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.RESUME, map_location='cpu')
    model = nn.DataParallel(model)
    msg = model.load_state_dict(checkpoint['model'])# , strict=False
    #logger.info(msg)
    max_accuracy = 0.0
    #optimizer=optimizer[0]
    #lr_scheduler=lr_scheduler[0]
    # import code
    # code.interact(local=locals())
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:  #config.phase == 'train' and
        # import code
        # code.interact(local=locals())
        #config.defrost()
        config.start_epoch = checkpoint['epoch']+1
        #config.freeze()
        logger.info(f"=> loaded successfully '{config.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy



@functools.lru_cache()
def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    #color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                #colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    # if dist_rank == 0:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    output = output_dir.split('-')
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log.txt'))#, mode='a'  # output[-2] + '_'+
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth') or ckpt.endswith('pt')]    #checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        # import code
        # code.interact(local=locals())
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file




class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()

  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    #correct_k = correct[:k].view(-1).float().sum(0)
    correct_k = correct[:k].contiguous().view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))

  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
  CIFAR_STD = [0.2675, 0.2565, 0.2761]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform




def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def _data_transforms_cifar(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  normalize_transform = [
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]

  random_transform = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()]

  if args.auto_aug:
    # import code
    # code.interact(local=locals())
    random_transform += [CIFAR10Policy()]

  if args.cutout:
    cutout_transform = [Cutout(args.cutout_length)]
  else:
    cutout_transform = []

  train_transform = transforms.Compose(
      random_transform + normalize_transform + cutout_transform
  )

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def _data_transforms_cifar_noaug(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  normalize_transform = [
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]

  random_transform = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()]

  if args.auto_aug:
    # import code
    # code.interact(local=locals())
    random_transform += [CIFAR10Policy()]

  if args.cutout:
    cutout_transform = [Cutout(args.cutout_length)]
  else:
    cutout_transform = []

  train_transform = transforms.Compose(
      normalize_transform + cutout_transform
  )

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform





def _data_transforms_cifar10_rotate(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
      transforms.RandomChoice(
          [
              transforms.RandomAffine(degrees=args.degree),
          ]),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform



def _data_transforms_cifar10_shear(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
      transforms.RandomChoice(
          [
              transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=args.degree),
              #transforms.RandomAffine(degrees=args.degree),
          ]),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp

class Temp_Scheduler_cifar(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_factor, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_factor=temp_factor
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        #self.total_epochs = total_epochs
        self.step()  #last_epoch + 1


    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        #self.total_epochs = 150
        self.curr_temp = self.base_temp*(self.temp_factor**self.last_epoch) + self.temp_min #(1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        # if self.curr_temp==5.0:
        #     import code
        #     code.interact(local=locals())
        return self.curr_temp

class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp

class DecayScheduler(object):
    def __init__(self, base_lr=1.0, last_iter=-1, T_max=50, decay_type='cosine'):
        self.base_lr = base_lr
        self.T_max = T_max
        self.T_start = 0
        self.T_stop = T_max
        self.cnt = 0
        self.decay_type = decay_type
        self.decay_rate = 1.0

    def step(self, epoch):
        if epoch >= self.T_start:
          if self.decay_type == "cosine":
              self.decay_rate = self.base_lr * (1 + math.cos(math.pi * epoch / (self.T_max - self.T_start))) / 2.0 if epoch <= self.T_stop else self.decay_rate
          elif self.decay_type == "slow_cosine":
              self.decay_rate = self.base_lr * math.cos((math.pi/2) * epoch / (self.T_max - self.T_start)) if epoch <= self.T_stop else self.decay_rate
          elif self.decay_type == "linear":
              self.decay_rate = self.base_lr * (self.T_max - epoch) / (self.T_max - self.T_start) if epoch <= self.T_stop else self.decay_rate
          else:
              self.decay_rate = self.base_lr
        else:

            self.decay_rate = self.base_lr
        return self.decay_rate



def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)



def load(config, model):
    checkpoint = torch.load(config.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return msg
    #model.load_state_dict(torch.load(model_path))

def load_pt(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)

def load_pt1(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def normalize(v):
  min_v = torch.min(v)
  range_v = torch.max(v) - min_v
  if range_v > 0:
    normalized_v = (v - min_v) / range_v
  else:
    normalized_v = torch.zeros(v.size()).cuda()

  return normalized_v

def histogram_intersection(a, b):
  c = np.minimum(a.cpu().numpy(),b.cpu().numpy())
  c = torch.from_numpy(c).cuda()
  sums = c.sum(dim=1)
  return sums


import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

		# redirect std err, if necessary

