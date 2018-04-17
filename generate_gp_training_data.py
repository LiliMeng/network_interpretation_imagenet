import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import cv2
import numpy as np
import random
from random import *
import sys

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0



def main():
    global args, best_prec1
    args = parser.parse_args()
    args.batch_size=1

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    count = 0
    for i, (input, target) in enumerate(val_loader):
        count += 1

        if count >10:
            break
        
        if count == 2:
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True).cuda()
            target_var = torch.autograd.Variable(target, volatile=True).cuda()

            org_img = input_var[0]
            org_img = org_img.type(torch.FloatTensor).data
            org_img = org_img.numpy()
            img = org_img.transpose( 1, 2, 0 )
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)

            # cv2.imshow('org_img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            segments = felzenszwalb(img_as_float(img), scale=100, sigma=0.5, min_size=100)
            
            print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))
            

            # cv2.imshow('superpixels', mark_boundaries(img_as_float(img), segments))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)


            pred = output.data.max(1, keepdim=True)[1]
            label = target[0]
            print("label ", label)
            print("pred[0].cpu().numpy() ", pred[0].cpu().numpy()[0])
           

            if pred[0].cpu().numpy()[0] == label:
                print("correct prediction, index ", count)

            correct_pred_count = 0
            wrong_pred_count = 0
            for i in range(1000): 
                num_conse_superpixels = 5
                total_num_segments = len(np.unique(segments))
                print("total_num_segments: ", total_num_segments)
                firstIndex= randint(1, total_num_segments-num_conse_superpixels)
               

                random_sampled_list = np.unique(segments)[firstIndex:(firstIndex + num_conse_superpixels)]              
                #random_sampled_list= random.sample(range(np.unique(segments)[0], np.unique(segments)[-1]), 5)
               
                print("random_sampled_list: ", random_sampled_list)
                mask = np.zeros(img.shape[:2], dtype= "uint8")
                #mask.fill(255)
                for (j, segVal) in enumerate(random_sampled_list):
                    mask[segments == segVal] = 255
                    

                masked_img_org = org_img * mask
                
                masked_img_org -= masked_img_org.min()
                masked_img_org /= masked_img_org.max()
                masked_img_org *= 255
                masked_img = normalize_image(masked_img_org)

                masked_img_batch = masked_img[None, :, :, :]

            
                masked_img_tensor = torch.autograd.Variable(torch.from_numpy(masked_img_batch)).cuda()
                mask_output = model(masked_img_tensor)
                
                pred_mask = mask_output.data.max(1, keepdim=True)[1]
               


                if pred_mask[0].cpu().numpy()[0] == target[0]:
                    correct_pred_count+=1
                    print("correct_pred_count: ", correct_pred_count)
                    cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 1), mask)
                    cv2.imwrite('./mask_on_img/masked_imgs_{}.png'.format(i), masked_img_org.transpose(1, 2, 0))
                else:
                    wrong_pred_count+=1
                    print("wrong_pred_count: ", wrong_pred_count)
                    cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 0), mask)
                    cv2.imwrite('./mask_on_img/masked_imgs_{}.png'.format(i), masked_img_org.transpose(1, 2, 0))

        #     # measure accuracy and record loss
        #     prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        #     losses.update(loss.data[0], input.size(0))
        #     top1.update(prec1[0], input.size(0))
        #     top5.update(prec5[0], input.size(0))

        #     # measure elapsed time
        #     batch_time.update(time.time() - end)
        #     end = time.time()

        #     if i % args.print_freq == 0:
        #         print('Test: [{0}/{1}]\t'
        #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #                i, len(val_loader), batch_time=batch_time, loss=losses,
        #                top1=top1, top5=top5))

        # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

def normalize_image(image):
    """Convert pixel intensity values from [0, 255] to [0.0, 1.0]."""
    return np.multiply(image.astype(np.float32), 1.0 / 255.0)


global args
args = parser.parse_args()

args.distributed = args.world_size > 1
args.batch_size=1

if args.distributed:
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size)

cudnn.benchmark = True

# create model
print("=> using pre-trained model '{}'".format(args.arch))
model = models.__dict__[args.arch](pretrained=True)

valdir = "/home/lili/Video/GP/examples/network_interpretation_imagenet/data/val"
#valdir = os.path.join(args.data, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

model.cuda()


# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
validate(val_loader, model, criterion)
 