#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
import cv2
import random
from colorama import Fore
from importlib import import_module
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import config
from dataloader import getDataloaders
from utils import save_checkpoint, get_optimizer, create_save_folder, normalize_image
from args import arg_parser, arch_resume_names



# Please refer to the following tutorial
#https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py


try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None

train_nn = False
prepare_GP_training_data = True
use_cuda = True

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

########################################################################
# CIFAR10 classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img, title_label):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title_label)
    plt.show()
    plt.close()


def getModel(arch, **kargs):
    m = import_module('models.' + arch)
    model = m.createModel(**kargs)
    if arch.startswith('alexnet') or arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model

def train_model():
    # parse arg and start experiment
    global args
    best_err1 = 100.
    best_epoch = 0

    args = arg_parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']
    if configure is None:
        args.tensorboard = False
        print(Fore.RED +
              'WARNING: you don\'t have tesnorboard_logger installed' +
              Fore.RESET)

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            old_args = checkpoint['args']
            print('Old args:')
            print(old_args)
            # set args based on checkpoint
            if args.start_epoch <= 0:
                args.start_epoch = checkpoint['epoch'] + 1
            best_epoch = args.start_epoch - 1
            best_err1 = checkpoint['best_err1']
            for name in arch_resume_names:
                if name in vars(args) and name in vars(old_args):
                    setattr(args, name, getattr(old_args, name))
            model = getModel(**vars(args))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print(
                "=> no checkpoint found at '{}'".format(
                    Fore.RED +
                    args.resume +
                    Fore.RESET),
                file=sys.stderr)
            return
    else:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = getModel(**vars(args))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = get_optimizer(model, args)

    # set random seed
    torch.manual_seed(args.seed)

    Trainer = import_module(args.trainer).Trainer
    trainer = Trainer(model, criterion, optimizer, args)

    # create dataloader
    if args.evaluate == 'train':
        train_loader, _, _ = getDataloaders(
            splits=('train'), **vars(args))
        trainer.test(train_loader, best_epoch)
        return
    elif args.evaluate == 'val':
        _, val_loader, _ = getDataloaders(
            splits=('val'), **vars(args))
        trainer.test(val_loader, best_epoch)
        return
    elif args.evaluate == 'test':
        _, _, test_loader = getDataloaders(
            splits=('test'), **vars(args))
        trainer.test(test_loader, best_epoch)
        return
    else:
        train_loader, val_loader, _ = getDataloaders(
            splits=('train', 'val'), **vars(args))

    # check if the folder exists
    create_save_folder(args.save, args.force)

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)
    log_print('args:')
    log_print(args)
    print('model:', file=f_log)
    print(model, file=f_log)
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))
    f_log.flush()
    torch.save(args, os.path.join(args.save, 'args.pth'))
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_err1'
              '\tval_err1\ttrain_err5\tval_err']
    if args.tensorboard:
        configure(args.save, flush_secs=5)

    for epoch in range(args.start_epoch, args.epochs + 1):

        # train for one epoch
        train_loss, train_err1, train_err5, lr = trainer.train(
            train_loader, epoch)

        if args.tensorboard:
            log_value('lr', lr, epoch)
            log_value('train_loss', train_loss, epoch)
            log_value('train_err1', train_err1, epoch)
            log_value('train_err5', train_err5, epoch)

        # evaluate on validation set
        val_loss, val_err1, val_err5 = trainer.test(val_loader, epoch)

        if args.tensorboard:
            log_value('val_loss', val_loss, epoch)
            log_value('val_err1', val_err1, epoch)
            log_value('val_err5', val_err5, epoch)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        scores.append(('{}\t{}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_err1, val_err1, train_err5, val_err5))
        with open(os.path.join(args.save, 'scores.tsv'), 'w') as f:
            print('\n'.join(scores), file=f)

        # remember best err@1 and save checkpoint
        is_best = val_err1 < best_err1
        if is_best:
            best_err1 = val_err1
            best_epoch = epoch
            print(Fore.GREEN + 'Best var_err1 {}'.format(best_err1) +
                  Fore.RESET)
            # test_loss, test_err1, test_err1 = validate(
            #     test_loader, model, criterion, epoch, True)
            # save test
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
        }, is_best, args.save)
        if not is_best and epoch - best_epoch >= args.patience > 0:
            break
    print('Best val_err1: {:.4f} at epoch {}'.format(best_err1, best_epoch))
	
def eval_superpixel():
     # parse arg and start experiment
    global args
    args = arg_parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']
    if configure is None:
        args.tensorboard = False
        print(Fore.RED +
              'WARNING: you don\'t have tesnorboard_logger installed' +
              Fore.RESET)

    model = getModel(**vars(args))
    saved_checkpoint = torch.load("./saved_checkpoints/cifar10+-resnet-56/model_best.pth.tar")
    model.load_state_dict(saved_checkpoint['state_dict'])
    
    model.eval()

    # get test images
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

    count = 0
    for images, labels in test_loader: 

        count +=1

        if count > 5:
            break
        # show images
       # imshow(torchvision.utils.make_grid(images), ' '.join('%5s' % classes[labels[j]] for j in range(1)))
        
        if use_cuda == True:
            images, labels = images.cuda(), labels.cuda()

        images, labels = Variable(images, volatile=True), Variable(labels)

        org_img = images[0]

        org_img = org_img.type(torch.FloatTensor).data
        org_img = org_img.numpy()
        img = org_img.transpose( 1, 2, 0 )
        img -= img.min()
        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
       
        # cv2.imshow('org_img_index{}_label_{}.png'.format(count, labels[0].cpu().data.numpy()[0]), img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
   
        if count == 5:

            cv2.imwrite('original_img_index{}_label_{}.png'.format(count, labels[0].cpu().data.numpy()[0]), img)

            segments = felzenszwalb(img_as_float(img), scale=100, sigma=0.5, min_size=10)
            
            print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))
            

            # cv2.imshow('superpixels', mark_boundaries(img_as_float(img), segments))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            output = model(images)
            pred = output.data.max(1, keepdim=True)[1]
            
       
            correct_pred_count = 0
            wrong_pred_count = 0
            for i in range(1000):               
                random_sampled_list= random.sample(range(np.unique(segments)[0], np.unique(segments)[-1]), 5)
               
                mask = np.zeros(img.shape[:2], dtype= "uint8")
                mask.fill(255)
                for (j, segVal) in enumerate(random_sampled_list):
                    mask[segments == segVal] = 0
                    

                masked_img = org_img * mask
                
                masked_img -= masked_img.min()
                masked_img /= masked_img.max()
                masked_img *= 255
                masked_img = normalize_image(masked_img)

                masked_img_batch = masked_img[None, :, :, :]

            
                masked_img_tensor = Variable(torch.from_numpy(masked_img_batch)).cuda()
                mask_output = model(masked_img_tensor)
                
                pred_mask = mask_output.data.max(1, keepdim=True)[1]
               
                print("pred_mask[0]", pred_mask[0].cpu().numpy()[0])

                if pred_mask[0].cpu().numpy()[0] == labels[0].cpu().data.numpy()[0]:
                    correct_pred_count+=1
                    print("correct_pred_count: ", correct_pred_count)
                    cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 1), mask)
                    cv2.imwrite('./mask_on_img/masked_imgs_{}.png'.format(i), masked_img.transpose(1, 2, 0))
                else:
                    wrong_pred_count+=1
                    print("wrong_pred_count: ", wrong_pred_count)
                    cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 0), mask)
                    cv2.imwrite('./mask_on_img/masked_imgs_{}.png'.format(i), masked_img.transpose(1, 2, 0))

          
                # plt.subplot(131),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'gray'),plt.title('org_img_label_{}'.format(classes[pred_mask[0].cpu().numpy()[0]]))
                # plt.subplot(132),plt.imshow(mark_boundaries(img_as_float(img), segments),'gray'),plt.title('Superpixel')
                # plt.subplot(133),plt.imshow(cv2.cvtColor(masked_img.transpose(1, 2, 0), cv2.COLOR_BGR2RGB),'gray'),plt.title('Org_img_with_mask_pred_{}'.format(classes[pred_mask[0].cpu().numpy()[0]]))
                # plt.show()
                # plt.close()
        
if train_nn == True:
  train_model()
elif prepare_GP_training_data == True:
  eval_superpixel()
else:
  raise Exception("Not implemented yet")
