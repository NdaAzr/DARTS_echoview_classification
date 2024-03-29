import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
#from torch.autograd import Variable
from model_search import Network
from architect import Architect
from datatrain import CustomDataset_classification, train_image_paths, train_tragets, classes, valid_image_paths, valid_targets 
from torch.utils.data import  SubsetRandomSampler

parser = argparse.ArgumentParser("cifar")
#parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate') #default=0.025
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay') # default=3e-4
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels') #default=16
parser.add_argument('--layers', type=int, default=8, help='total number of layers') #default=8
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed') #2
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
#parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding') #3e-4
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding') #1e-3
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#CIFAR_CLASSES = 10
CIFAR_CLASSES = 15

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum, weight_decay=args.weight_decay)
       
#  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = CustomDataset_classification(train_image_paths, train_tragets, classes, train_image_paths)

  num_train = len(train_data)
  indices = list(range(num_train))
#  split = int(np.floor(args.train_portion * num_train))
#dataset is splitted in datatrain module 
  train_queue = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,
                pin_memory=True, num_workers=2)
  
  valid_data = CustomDataset_classification(valid_image_paths, valid_targets, classes, valid_image_paths)  
  valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, 
                shuffle=False, pin_memory=True, num_workers=2)
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
#  lr = args.learning_rate
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr) #, lr
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr): #, lr
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (t_image, target, i, ii) in enumerate(train_queue):
    model.train()
    n = t_image.size(0)

    t_image = t_image.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search, i, ii = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()

    architect.step(t_image, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(t_image)
    #print(logits)
    loss = criterion(logits, target)

    loss.backward()
#    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss, n)
    top1.update(prec1, n)
    top5.update(prec5, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():

      for step, (t_image, target, i, ii) in enumerate(valid_queue):
                  
        t_image = t_image.cuda()
        target = target.cuda()
    
        logits = model(t_image)
        loss = criterion(logits, target)
    
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = t_image.size(0)
        objs.update(loss, n)
        top1.update(prec1, n)
        top5.update(prec5, n)
    
        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    
      return top1.avg, objs.avg    

if __name__ == '__main__':
  main() 

