import cv2
import numpy as np
import random
from random import *
import sys
import time

import matplotlib.pyplot as plt
from operator import itemgetter
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


from utils import normalize_image
from imagenet_lables import *
from BayesianOptimization import *

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
import torch.nn.functional as F

best_prec1 = 0
np.set_printoptions(threshold=np.nan)


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
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
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
parser.add_argument('--eval_img_index', default=400, type=int,
                    help='the index of evaluation image')
parser.add_argument('--num_mask_samples', default=1, type=int,
                    help='the number of mask samples')

dataset = "imagenet"
if dataset == "imagenet":
    n = 224
else:
    raise Exception("this dataset is not implemented yet")

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


def validate_nueral_network(val_loader, model, criterion, eval_img_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
        
    count = 0
    for i, (input, target) in enumerate(val_loader):
        count += 1

        if count > eval_img_index:
            break
            
        input_var = torch.autograd.Variable(input.cuda(), requires_grad=True)
        input_var.requires_grad = True
        target_var = torch.autograd.Variable(target).cuda()
        
        img_show = input[0].numpy().copy()
        img_show = img_show.transpose( 1, 2, 0 )

     
        img_show -= img_show.min()
        img_show /= img_show.max()
        img_show *= 255
        img_show = img_show.astype(np.uint8)
       
        # cv2.imshow('index_{}_label_{}'.format(i, classes_dict[target[0]]), img_show)
        # # cv2.waitKey(0)
        if count == eval_img_index:
            cv2.imwrite("org_img.png", img_show)
            
            segments = felzenszwalb(img_as_float(img_show), scale=100, sigma=0.5, min_size=50)
                        
            print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))
            
            print("segments")
            print(segments)

            #cv2.imshow('superpixels', mark_boundaries(img_as_float(img_show), segments))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            pred = output.data.max(1, keepdim=True)[1]
            label = target[0]
            print("label ", label)
            print("pred[0].cpu().numpy() ", pred[0].cpu().numpy()[0])

            mask_dir = './masks'
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            else:
                shutil.rmtree(mask_dir)           #removes all the subdirectories!
                os.makedirs(mask_dir)
           

            if pred[0].cpu().numpy()[0] == label:
                print("correct prediction, index_{} , label_{}".format(count, classes_dict[label]))

                correct_pred_count = 0
                wrong_pred_count = 0

                summed_superpixel_labels = np.zeros(img_show.shape[:2])

                
                for i in range(args.num_mask_samples): 
                    correct_label_flag = False
                    print("np.unique(segments)")
                    print(np.unique(segments))
                    total_num_segments = len(np.unique(segments))
                    num_conse_superpixels = int(0.4*total_num_segments)
                    print("total_num_segments: ", total_num_segments)
                    print("num_conse_superpixels: ", num_conse_superpixels)
                    firstIndex= randint(1, total_num_segments-num_conse_superpixels)
                   

                    random_sampled_list = np.unique(segments)[firstIndex:(firstIndex + num_conse_superpixels)]              
                    #random_sampled_list= sample(range(np.unique(segments)[0], np.unique(segments)[-1]), num_conse_superpixels)
                    segments_unique = np.unique(segments)

                    mask = np.zeros(img_show.shape[:2], dtype= "uint8")
                    #mask.fill(1)
                    for (j, segVal) in enumerate(random_sampled_list):
                        mask[segments == segVal] = 1
                        
                    
                    masked_img = input[0].numpy().copy() * mask

                    masked_img_batch = masked_img[None, :, :, :]

                
                    masked_img_tensor = torch.autograd.Variable(torch.from_numpy(masked_img_batch)).cuda()
                    mask_output = model(masked_img_tensor)
                    
                    probability_score = np.exp(mask_output.data)
                    pred_mask = mask_output.data.max(1, keepdim=True)[1]
                    probability_score = F.softmax(mask_output)

                    
                    print("probability_score: ", probability_score)
                   
                    masked_img_show = masked_img.copy()
                    masked_img_show = masked_img_show.transpose(1, 2, 0)
                    masked_img_show -= masked_img_show.min()
                    masked_img_show /= masked_img_show.max()
                    masked_img_show *= 255
                    masked_img_show = masked_img_show.astype(np.uint8)

                    max_prob = torch.max(probability_score).data.cpu().numpy()[0]

                    print("max_prob: ", max_prob)

                    if pred_mask[0].cpu().numpy()[0] == target[0]:
                        correct_pred_count+=1
                        print("correct_pred_count: ", correct_pred_count)
                        cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 1), mask*255) 
                        correct_label_flag = True

                        return max_prob, correct_label_flag
                    else:
                        wrong_pred_count+=1
                        print("wrong_pred_count: ", wrong_pred_count)
                        cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 0), mask*255)
                        
                        return max_prob, correct_label_flag
            else:
                print("wrong prediction")
                #print("%d samples, the corrrect prediction number: %d "%(len(mask_filenames), correct_pred_count))
                return -1, False       

def sample_loss(params):
    """
        The loss for each sample in objective function. 
        softmax probability with a regularizer to constrain the superpixel size 
    """
    start_time = time.time()
    global args
    args = parser.parse_args()

    args.distributed = args.world_size > 1
    args.batch_size=1

    val_data_dir = "/home/lili/Video/GP/examples/network_interpretation_imagenet/data/val"

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    model.cuda()

    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_data_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    eval_img_index = 1600
    superpixel_percent = 0.4

    max_prob, correct_pred_flag = validate_nueral_network(val_loader, model, criterion, eval_img_index)
    
    regularizer = 0.01
    sample_loss_value = 0 
    if max_prob!=-1:
        if correct_pred_flag == True:
            
            sample_loss_value = max_prob + regularizer*superpixel_percent

    return sample_loss_value
    

def main():

    
    bounds = np.array([[-4, 1], [-4, 1]])

    xp, yp = bayesian_optimisation(n_iters=30, 
                               sample_loss=sample_loss, 
                               bounds=bounds,
                               n_pre_samples=3,
                               random_search=100000)

    time_duration = time.time()-start_time

    print("time duration is: ", time_duration) 

    

if __name__== "__main__":
  main()

