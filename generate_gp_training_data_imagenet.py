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

import matplotlib.pyplot as plt
from operator import itemgetter
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from utils import normalize_image
from imagenet_lables import *


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
parser.add_argument('--num_mask_samples', default=100, type=int,
                    help='the number of mask samples')


dataset = "imagenet"
if dataset == "imagenet":
    n = 224
else:
    raise Exception("this dataset is not implemented yet")

# Training data
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


def validate(val_loader, model, criterion, eval_img_index):
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
            
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()
        
        img_show = input[0].numpy().copy()
        img_show = img_show.transpose( 1, 2, 0 )

     
        img_show -= img_show.min()
        img_show /= img_show.max()
        img_show *= 255
        img_show = img_show.astype(np.uint8)
        # cv2.imshow('index_{}_label_{}'.format(i, classes_dict[target[0]]), img_show)
    #     # cv2.waitKey(0)
        if count == eval_img_index:
            
            segments = felzenszwalb(img_as_float(img_show), scale=100, sigma=0.5, min_size=50)
            
            print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))
            

            #cv2.imshow('superpixels', mark_boundaries(img_as_float(img_show), segments))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            print("output")
            print(output)
            print("target_var")
            print(target_var)
            print("loss")
            print(loss)

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

                for i in range(args.num_mask_samples): 
                    
                    total_num_segments = len(np.unique(segments))
                    num_conse_superpixels = int(0.4*total_num_segments)
                    print("total_num_segments: ", total_num_segments)
                    print("num_conse_superpixels: ", num_conse_superpixels)
                    firstIndex= randint(1, total_num_segments-num_conse_superpixels)
                   

                    random_sampled_list = np.unique(segments)[firstIndex:(firstIndex + num_conse_superpixels)]              
                    #random_sampled_list= sample(range(np.unique(segments)[0], np.unique(segments)[-1]), num_conse_superpixels)
                   
                    #print("random_sampled_list: ", random_sampled_list)
                    mask = np.zeros(img_show.shape[:2], dtype= "uint8")
                    #mask.fill(1)
                    for (j, segVal) in enumerate(random_sampled_list):
                        mask[segments == segVal] = 1
                        
                    
                    masked_img = input[0].numpy().copy() * mask

                    masked_img_batch = masked_img[None, :, :, :]

                
                    masked_img_tensor = torch.autograd.Variable(torch.from_numpy(masked_img_batch)).cuda()
                    mask_output = model(masked_img_tensor)
                    
                    pred_mask = mask_output.data.max(1, keepdim=True)[1]
                   
                    masked_img_show = masked_img.copy()
                    masked_img_show = masked_img_show.transpose(1, 2, 0)
                    masked_img_show -= masked_img_show.min()
                    masked_img_show /= masked_img_show.max()
                    masked_img_show *= 255
                    masked_img_show = masked_img_show.astype(np.uint8)

                    if pred_mask[0].cpu().numpy()[0] == target[0]:
                        correct_pred_count+=1
                        print("correct_pred_count: ", correct_pred_count)
                        cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 1), mask*255)
                       # cv2.imwrite('./mask_on_img/masked_imgs_{}.png'.format(i), masked_img_show)
                    else:
                        wrong_pred_count+=1
                        print("wrong_pred_count: ", wrong_pred_count)
                        cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 0), mask*255)
                        #cv2.imwrite('./mask_on_img/masked_imgs_{}.png'.format(i), masked_img_show)
                
                return correct_pred_count
            else:
                print("wrong prediction")
                #print("%d samples, the corrrect prediction number: %d "%(len(mask_filenames), correct_pred_count))
                
    return 0

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

# Training data
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


def validate_mask(val_loader, model, criterion, val_img_index):

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

        if count > val_img_index:
            break
        
        
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        
        img_show = input[0].numpy().copy()
        img_show = img_show.transpose( 1, 2, 0 )
        img_show -= img_show.min()
        img_show /= img_show.max()
        img_show *= 255
        img_show = img_show.astype(np.uint8)

        if count == val_img_index:
            #cv2.imwrite("orginal_img.png", img_show)
           
            segments = felzenszwalb(img_as_float(img_show), scale=100, sigma=0.5, min_size=50)
            
            print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))
            

            # cv2.imshow('superpixels', mark_boundaries(img_as_float(img_show), segments))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)


            pred = output.data.max(1, keepdim=True)[1]
            label = target[0]

            if pred[0].cpu().numpy()[0] == label:
                print("correct prediction, index ", count)

            correct_pred_count = 0
            wrong_pred_count = 0
                
            dict_pixel = get_pixel_sorted_mask_label()
            plot_summed_heatmap(val_img_index, img_show, label, dict_pixel)
            sorted_dict_values_set = sorted(set(dict_pixel.values())) 

            # Binary search the pixel label threshold
            first = 0
            last = len(sorted_dict_values_set) -1 

            while first<= last:

                midpoint = int((first + last)/2)
            
                mask_threshold1 = sorted_dict_values_set[midpoint]

                mask_threshold2 = sorted_dict_values_set[midpoint+1]   
                
                mask1 = generate_new_mask(dict_pixel, mask_threshold1)

                mask2 = generate_new_mask(dict_pixel, mask_threshold2)

                print("sorted_dict_values_set")
                print(sorted_dict_values_set)  

                print("len(sorted_dict_values_set)")
                print(len(sorted_dict_values_set))  
                          
                masked_img1 = input[0].numpy().copy() * mask1
                masked_img2 = input[0].numpy().copy() * mask2
                
                masked_img_batch1 = masked_img1[None, :, :, :]
                masked_img_batch2 = masked_img2[None, :, :, :]

                masked_img_tensor1 = torch.autograd.Variable(torch.from_numpy(masked_img_batch1)).cuda()
                masked_img_tensor2 = torch.autograd.Variable(torch.from_numpy(masked_img_batch2)).cuda()
                # Evaluate the NN model 
                mask_output1 = model(masked_img_tensor1)
                mask_output2 = model(masked_img_tensor2)
                
                pred_mask1 = mask_output1.data.max(1, keepdim=True)[1]
                pred_mask2 = mask_output2.data.max(1, keepdim=True)[1]


                masked_img_show1 = masked_img1.copy()
                masked_img_show1 = masked_img_show1.transpose(1, 2, 0)
                masked_img_show1 -= masked_img_show1.min()
                masked_img_show1 /= masked_img_show1.max()
                masked_img_show1 *= 255
                masked_img_show1 = masked_img_show1.astype(np.uint8)
               
                masked_img_show2 = masked_img2.copy()
                masked_img_show2 = masked_img_show2.transpose(1, 2, 0)
                masked_img_show2-= masked_img_show2.min()
                masked_img_show2 /= masked_img_show2.max()
                masked_img_show2 *= 255
                masked_img_show2 = masked_img_show2.astype(np.uint8)
            
                if pred_mask1[0].cpu().numpy()[0] == target[0]:

                    correct_pred_count+=1
                    print("correct_pred_count: ", correct_pred_count)
                    
                    if pred_mask2[0].cpu().numpy()[0]!= target[0]:
            
                        plt.subplot(141),plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB), 'gray'),plt.title('original_img_label_{}'.format(classes_dict[target[0]]),  fontsize=60)
                        plt.subplot(142),plt.imshow(mark_boundaries(img_as_float(img_show[:,:,::-1]), segments),'gray'),plt.title('Superpixel',  fontsize=60)
                      
                        plt.subplot(143),plt.imshow(mask1*255,  'gray'), plt.title("Mask threshold_{}".format(mask_threshold1),  fontsize=60)
                        plt.subplot(144),plt.imshow(cv2.cvtColor(masked_img_show1, cv2.COLOR_BGR2RGB),'gray'),plt.title('Org_img_with_mask pred_{}'.format(classes_dict[pred_mask1[0].cpu().numpy()[0]]),  fontsize=60)
                        
                        #cv2.imwrite("frog.png", masked_img_show1)
                        figure = plt.gcf() # get current figure
                        figure.set_size_inches(90, 30)
                      
                        plt.savefig('result_imgs/index_{}_threshold_{}.png'.format(val_img_index, mask_threshold1))
                                          
                        plt.subplot(141),plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB), 'gray'),plt.title('original_img_label_{}'.format(classes_dict[target[0]]), fontsize=60)
                        plt.subplot(142),plt.imshow(mark_boundaries(img_as_float(img_show[:,:,::-1]), segments),'gray'),plt.title('Superpixel', fontsize=60)
                        
                        plt.subplot(143),plt.imshow(mask2*255,  'gray'), plt.title("Mask threshold_{}".format(mask_threshold2), fontsize=60)
                        plt.subplot(144),plt.imshow(cv2.cvtColor(masked_img_show2, cv2.COLOR_BGR2RGB),'gray'),plt.title('pred_{}'.format(classes_dict[pred_mask2[0].cpu().numpy()[0]]),fontsize=60)
                        
                        figure = plt.gcf() # get current figure
                        figure.set_size_inches(90, 30)
                        plt.savefig('result_imgs/index_{}_threshold_{}.png'.format(val_img_index, mask_threshold2))
                        # plt.show()
                        # plt.close()
                 
                        return mask_threshold1
                    else:
                        first = midpoint+1      
                else:
                    wrong_pred_count+=1
                    print("wrong_pred_count: ", wrong_pred_count)
                    last = midpoint-1
                    print("masked label threshold")
                    print(mask_threshold1)
                        
                    
                          
def get_pixel_sorted_mask_label():
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
   
    print("%d samples, the corrrect prediction number: %d "%(len(mask_filenames), correct_pred_count))
   
    return dict_pixel

def plot_summed_heatmap(val_img_index, org_img, label, dict_pixel):

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
        
    plt.subplot(121),plt.imshow(org_img[:,:,::-1],'gray'),plt.title('Org_img_with_label_{}'.format(classes_dict[label]), fontsize=60)

    plt.subplot(122),plt.imshow(result_heatmap[:,:,::-1],'gray'),plt.title('Summed label training heatmap', fontsize=60)
    
    figure = plt.gcf() # get current figure
    figure.set_size_inches(80, 30)
                          
    plt.colorbar()
    plt.set_cmap('jet')
    plt.savefig('result_imgs/index_{}_label_{}.png'.format(val_img_index, classes_dict[label]))


def generate_new_mask(dict_pixel, mask_threshold):

    result_gray_img = np.zeros((n,n))
    result_mask = np.zeros((n, n), dtype= "uint8")
    for i in range(n):
        for j in range(n):
            pixel_pos = (i,j)
            if pixel_pos in dict_pixel:
                if dict_pixel[pixel_pos] <= mask_threshold:
                    result_gray_img[i][j] = 0
                    result_mask[i][j] = 0
                else:
                    result_gray_img[i][j] = dict_pixel[pixel_pos] 
                    result_mask[i][j] = 1

    return result_mask

def main():

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

    eval_img_index = 300

    summed_superpixel_labels, summed_superpixel_labels_heatmap = validate(val_loader, model, criterion, eval_img_index)


  
if __name__== "__main__":
  main()

