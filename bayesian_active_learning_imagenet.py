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
from utils import generate_boundingbox, generate_IOU
from dataset import imagenet_localization_dataset


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
parser.add_argument('--eval_img_index', default=1600, type=int,
                    help='the index of evaluation image')
parser.add_argument('--num_mask_samples', default=1, type=int,
                    help='the number of mask samples')

dataset = "imagenet"
if dataset == "imagenet":
    n = 224
else:
    raise Exception("this dataset is not implemented yet")

global args
args = parser.parse_args()

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


def validate_nueral_network(val_loader, model, criterion, bo_iter, firstIndex):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
        
    count = 0
    for i, (input, target, gt_bboxes) in enumerate(val_loader):
        count += 1

        if count > args.eval_img_index:
            break
            
        input_var = torch.autograd.Variable(input.cuda(), requires_grad=True)
        input_var.requires_grad = True
        target_var = torch.autograd.Variable(target[0]).cuda()
        
        img_show = input[0].numpy().copy()
        img_show = img_show.transpose( 1, 2, 0 )

     
        img_show -= img_show.min()
        img_show /= img_show.max()
        img_show *= 255
        img_show = img_show.astype(np.uint8)
       
        # cv2.imshow('index_{}_label_{}'.format(i, classes_dict[target[0]]), img_show)
        # # cv2.waitKey(0)
        if count == args.eval_img_index:
            cv2.imwrite("org_img.png", img_show)
            
            segments = felzenszwalb(img_as_float(img_show), scale=100, sigma=0.5, min_size=50)
                        
            print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))

            #cv2.imshow('superpixels', mark_boundaries(img_as_float(img_show), segments))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # compute output
            output = model(input_var)

            loss = criterion(output, target_var)

            pred = output.data.max(1, keepdim=True)[1]
            label = target[0].numpy()[0]
          

            if pred[0].cpu().numpy()[0] == label:
                print("correct prediction, index_{} , label_{}".format(count, classes_dict[label]))

                correct_pred_count = 0
                wrong_pred_count = 0
                   
                total_num_segments = len(np.unique(segments))
                num_conse_superpixels = int(0.4*total_num_segments)
                print("total_num_segments: ", total_num_segments)
                print("num_conse_superpixels: ", num_conse_superpixels)
               
                random_sampled_list = np.unique(segments)[firstIndex:(firstIndex + num_conse_superpixels)]              
                #random_sampled_list= sample(range(np.unique(segments)[0], np.unique(segments)[-1]), num_conse_superpixels)
                segments_unique = np.unique(segments)

                mask = np.zeros(img_show.shape[:2], dtype= "uint8")
         
                for (j, segVal) in enumerate(random_sampled_list):
                    mask[segments == segVal] = 1
                    
                masked_img = input[0].numpy().copy() * mask

                masked_img_batch = masked_img[None, :, :, :]
            
                masked_img_tensor = torch.autograd.Variable(torch.from_numpy(masked_img_batch)).cuda()
                mask_output = model(masked_img_tensor)
                
            
                pred_mask = mask_output.data.max(1, keepdim=True)[1]
                total_probability_score = F.softmax(mask_output)
                
                class_prob_score = total_probability_score.data.cpu().numpy()[0][label]

                masked_img_show = masked_img.copy()
                masked_img_show = masked_img_show.transpose(1, 2, 0)
                masked_img_show -= masked_img_show.min()
                masked_img_show /= masked_img_show.max()
                masked_img_show *= 255
                masked_img_show = masked_img_show.astype(np.uint8)

                if pred_mask[0].cpu().numpy()[0] == label:
                    correct_pred_count+=1
                    print("correct_pred_count: ", correct_pred_count)
                    cv2.imwrite('./masks/mask_{}_{}.png'.format(bo_iter, 1), mask*255) 
                    cv2.imwrite('./mask_on_img/masked_imgs_{}_{}.png'.format(bo_iter,1), masked_img_show)            
                else:                  
                    wrong_pred_count+=1
                    print("wrong_pred_count: ", wrong_pred_count)
                    cv2.imwrite('./masks/mask_{}_{}.png'.format(bo_iter, 0), mask*255)
                    cv2.imwrite('./mask_on_img/masked_imgs_{}_{}.png'.format(bo_iter, 0), masked_img_show)
                    
                return class_prob_score
            else: 
                print("wrong prediction")
                raise Exception("currently this situation is not considered yet")     


def superpixel_mask(firstIndex):

    val_data_dir = "/home/lili/Video/GP/examples/network_interpretation_imagenet/data/val"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(imagenet_localization_dataset(
                    data_dir=val_data_dir,
                    crop = -1,
                    transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
        batch_size = 1, shuffle=False,
        num_workers = 1, pin_memory=True)

    count = 0

    for i, (input, target, gt_bboxes) in enumerate(val_loader):
        count += 1

        if count > args.eval_img_index:
            break
            
        input_var = torch.autograd.Variable(input.cuda(), requires_grad=True)
        input_var.requires_grad = True
        target_var = torch.autograd.Variable(target).cuda()

        if count == args.eval_img_index:
            img_show = input[0].numpy().copy()
            img_show = img_show.transpose( 1, 2, 0 )
            img_show -= img_show.min()
            img_show /= img_show.max()
            img_show *= 255
            img_show = img_show.astype(np.uint8)
            
            segments = felzenszwalb(img_as_float(img_show), scale=100, sigma=0.5, min_size=50)
            total_num_segments = len(np.unique(segments))
            num_conse_superpixels = int(0.4*total_num_segments)

            random_sampled_list = np.unique(segments)[firstIndex:(firstIndex + num_conse_superpixels)]              
        
            segments_unique = np.unique(segments)

            mask = np.zeros(img_show.shape[:2], dtype= "uint8")
             
            for (j, segVal) in enumerate(random_sampled_list):
                mask[segments == segVal] = 1
                        
            return mask*255
              
def sample_loss(params, val_loader, model, criterion):
    """
        The loss for each sample in objective function. 
        softmax probability with a regularizer to constrain the superpixel size 
    """
    print("params")
    print(params)
    firstIndex = int(params[0])

    bo_iter = params[0]

    print("firstIndex: ", firstIndex)

    superpixel_percent = 0.4

    class_prob_score = validate_nueral_network(val_loader, model, criterion, bo_iter, firstIndex)
    
    regularizer = 0.01
    sample_loss_value = 0 

    sample_loss_value = class_prob_score # + regularizer*superpixel_percent

    return sample_loss_value

def load_images_from_folder(folder):
    img_filenames = []
    labels = []
    for filename in os.listdir(folder):      
        label=filename.split('_')[2].split('.')[0]
        img_filename = os.path.join(folder,filename)
        if img_filename is not None:
            img_filenames.append(img_filename)
            labels.append(label)
    return img_filenames, labels


def plot_summed_heatmap(val_img_index, bbox_threshold, gt_bbox):
    mask_filenames, train_mask_labels = load_images_from_folder('./masks')

    train_x = []
    train_y = []
    pixel_mask_counts = []
    dict_pixel = {}

    correct_pred_count = 0
    for i in range(len(mask_filenames)):
        img = cv2.imread(mask_filenames[i] ,0)
        mask_label = int(train_mask_labels[i])
        if mask_label == 1:
            correct_pred_count +=1
        print('has read ', i)
        for j in range(n):
            for k in range(n):
                pixel_position = (j, k)        
                if img[j][k] == 255:
                    if pixel_position in dict_pixel:
                        dict_pixel[pixel_position] += mask_label
                    else:
                        dict_pixel[pixel_position]  = mask_label
   
    print("%d samples, the correct prediction number: %d "%(len(mask_filenames), correct_pred_count))

    result_gray_img = np.zeros((n,n))
    result_mask = np.zeros((n, n), dtype= "uint8")
    for i in range(n):
        for j in range(n):
            pixel_pos = (i,j)
            if pixel_pos in dict_pixel:
                result_gray_img[i][j] = dict_pixel[pixel_pos] 
                

    result_gray_img_show = result_gray_img.copy()

    result_gray_img_show = result_gray_img_show - result_gray_img_show.min()
    result_gray_img_show = result_gray_img_show/result_gray_img_show.max()
    result_gray_img_show *= 255

    result_gray_img_show = np.array(result_gray_img_show, dtype = np.uint8)
    result_heatmap = cv2.applyColorMap(result_gray_img_show, cv2.COLORMAP_JET)

    org_img = cv2.imread('org_img.png')
        
    plt.subplot(121),plt.imshow(org_img[:,:,::-1],'gray'),plt.title('Org_img', fontsize=60)

    plt.subplot(122),plt.imshow(result_heatmap[:,:,::-1],'gray'),plt.title('Summed label training heatmap', fontsize=60)
    plt.set_cmap('jet')
    plt.colorbar()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(80, 30)
    
    plt.savefig('result_imgs/index_{}.png'.format(val_img_index))
    cv2.imwrite("heatmaps/index_{}.png".format(val_img_index), result_heatmap)
    
    pred_box = generate_boundingbox(val_img_index, result_heatmap, bbox_threshold, "heatmaps")

    IOU = generate_IOU(pred_box, gt_bbox)
    print('\033[91m' + "IOU: " + str(IOU) + '\033[0m')

def main():

 
    start_time = time.time()

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

    val_loader = torch.utils.data.DataLoader(imagenet_localization_dataset(
                    data_dir=val_data_dir,
                    crop = -1,
                    transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
        batch_size = 1, shuffle=False,
        num_workers = 1, pin_memory=True)

    #val_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(val_data_dir, transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])),
    #    batch_size=args.batch_size, shuffle=False,
    #    num_workers=args.workers, pin_memory=True)
  
    # # The Oracle
    # param_grid = np.array([C for C in range(44)])

    # real_loss = [sample_loss(params, val_loader, model, criterion) for params in param_grid]

    # plt.figure()
    # plt.plot(param_grid, real_loss, 'go--', linewidth=2, markersize=12)
    # plt.show()
    count = 0
    for i, (input, target, gt_bboxes) in enumerate(val_loader):
        count += 1

        if count > args.eval_img_index:
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
        if count == args.eval_img_index:

            x, y, w, h = gt_bboxes[0]

            cv2.imwrite("org_img.png", img_show)
            
            segments = felzenszwalb(img_as_float(img_show), scale=100, sigma=0.5, min_size=50)
                        
            print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))

            firstIndex_upperbound = int(0.6*len(np.unique(segments)))

            print("firstIndex_upperbound: ", firstIndex_upperbound)
    
            mask_dir = './masks'
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            else:
                shutil.rmtree(mask_dir)           
                os.makedirs(mask_dir)

            bounds = np.asarray([[0, firstIndex_upperbound]])
            xp, yp = bayesian_optimisation(n_iters=10, 
                                        sample_loss=sample_loss, 
                                        val_loader = val_loader,
                                        nn_model = model,
                                        criterion = criterion,
                                        bounds=bounds,
                                        n_pre_samples=3,
                                        random_search=False)

            time_duration = time.time()-start_time

            print("time duration is: ", time_duration) 
            bbox_threshold = 160
            plot_summed_heatmap(args.eval_img_index, bbox_threshold, gt_bboxes[0])
    



if __name__== "__main__":
  main()


