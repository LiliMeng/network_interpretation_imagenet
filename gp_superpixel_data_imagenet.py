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

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import cv2
import os

from torch.autograd import Variable

import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import scipy.misc
from torch import nn, optim
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.random_variables import GaussianRandomVariable

from utils import normalize_image

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import cv2
import os

from torch.autograd import Variable


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
        # # cv2.waitKey(0)
        if count == eval_img_index:
            
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
                    
                    total_num_segments = len(np.unique(segments))
                    num_conse_superpixels = int(0.4*total_num_segments)
                    print("total_num_segments: ", total_num_segments)
                    print("num_conse_superpixels: ", num_conse_superpixels)
                    firstIndex= randint(1, total_num_segments-num_conse_superpixels)
                   

                    random_sampled_list = np.unique(segments)[firstIndex:(firstIndex + num_conse_superpixels)]              
                    #random_sampled_list= sample(range(np.unique(segments)[0], np.unique(segments)[-1]), num_conse_superpixels)
                    segments_unique = np.unique(segments)
                    print("segments_unique")
                    print(segments_unique)
                    print("random_sampled_list")
                    print(random_sampled_list)
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


                        for (j, segVal) in enumerate(random_sampled_list):
                            summed_superpixel_labels[segments == segVal] +=1
                            print("summed_superpixel_labels")
                            print(summed_superpixel_labels)

                        
                       # cv2.imwrite('./mask_on_img/masked_imgs_{}.png'.format(i), masked_img_show)
                    else:
                        wrong_pred_count+=1
                        print("wrong_pred_count: ", wrong_pred_count)
                        cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 0), mask*255)
                        #cv2.imwrite('./mask_on_img/masked_imgs_{}.png'.format(i), masked_img_show)
                
                print("summed_superpixel_labels")
                print(summed_superpixel_labels)
                #summed_superpixel_labels_show = summed_superpixel_labels.copy()[:,:,None]
                #print("summed_superpixel_labels_show.shape: ", summed_superpixel_labels_show.shape)
                #summed_superpixel_labels_show -= summed_superpixel_labels_show.min()
                #summed_superpixel_labels_show /= summed_superpixel_labels_show.max()
                #summed_superpixel_labels_show *= 255
                #summed_superpixel_labels_show = summed_superpixel_labels_show.astype(np.uint8)
                #summed_labels_heatmap = cv2.applyColorMap(summed_superpixel_labels_show, cv2.COLORMAP_JET)
                #cv2.imwrite("summed_superpixel_labels.png", summed_labels_heatmap)
                #cv2.imshow("summed_superpixel_labels", summed_labels_heatmap)

                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                return summed_superpixel_labels
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

                   
# We use KISS-GP (kernel interpolation for scalable structured Gaussian Processes)
# as in https://arxiv.org/pdf/1503.01057.pdf
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        # Near-zero mean
        self.mean_module = ConstantMean(constant_bounds=[-1e-5,1e-5])
        # GridInterpolationKernel over an ExactGP
        self.base_covar_module = RBFKernel(log_lengthscale_bounds=(-5, 6))
        self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=30,
                                                    grid_bounds=[(0, n), (0, n)])
        # Register the log lengthscale as a trainable parametre
        self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-5,6))
        
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        return GaussianRandomVariable(mean_x, covar_x)


def train(train_x, train_y, model, optimizer, mll):
    num_training_iterations = 20
    
    for i in range(num_training_iterations):
        
        batch_training = False

        if batch_training == True:
            train_loss = 0
            train_batch_size = 100000
            full_output = []
            for j in range(0, train_x.shape[0], train_batch_size):
                train_indices = range(train_x.shape[0])[j: j+ train_batch_size]

                train_batch_x = train_x[train_indices]
                train_batch_y = train_y[train_indices]

                # zero back propped gradients
                optimizer.zero_grad()

                # Make  prediction
                output_batch = model(train_batch_x)

                # Calc loss and use to compute derivatives
                loss = -mll(output_batch, train_batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.data[0] * len(train_batch_x)
            print('Train Epoch: %d\tLoss: %.6f' % (i, train_loss / train_x.shape[0]))

        else:

            # zero back propped gradients
            optimizer.zero_grad()
            # Make  prediction
            output = model(train_x)

            # Calc loss and use to compute derivatives
            loss = -mll(output, train_y)

            print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f' % (
                i + 1, num_training_iterations, loss.data[0],
                model.covar_module.base_kernel_module.log_lengthscale.data.squeeze()[0],
            ))
      

    torch.save(model.state_dict(), './gp_saved_checkpoints/imagenet100_gp_reg_checkpoint.pth.tar')


def train(train_x, train_y, model, optimizer, mll):
    num_training_iterations = 20
    for i in range(num_training_iterations):
        
        batch_training = False

        if batch_training == True:
            train_loss = 0
            train_batch_size = 100000
            full_output = []
            for j in range(0, train_x.shape[0], train_batch_size):
                train_indices = range(train_x.shape[0])[j: j+ train_batch_size]

                train_batch_x = train_x[train_indices]
                train_batch_y = train_y[train_indices]

                # zero back propped gradients
                optimizer.zero_grad()

                # Make  prediction
                output_batch = model(train_batch_x)

                # Calc loss and use to compute derivatives
                loss = -mll(output_batch, train_batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.data[0] * len(train_batch_x)
            print('Train Epoch: %d\tLoss: %.6f' % (i, train_loss / train_x.shape[0]))

        else:

            # zero back propped gradients
            optimizer.zero_grad()
            # Make  prediction
            output = model(train_x)

            # Calc loss and use to compute derivatives
            loss = -mll(output, train_y)

            print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f' % (
                i + 1, num_training_iterations, loss.data[0],
                model.covar_module.base_kernel_module.log_lengthscale.data.squeeze()[0],
            ))
      

    torch.save(model.state_dict(), './gp_saved_checkpoints/imagenet1000_gp_reg_checkpoint.pth.tar')

def eval_superpixels(model, likelihood):

    # load model
    model.load_state_dict(torch.load('./gp_saved_checkpoints/imagenet1000_gp_reg_checkpoint.pth.tar'))
    # Set model and likelihood into eval mode
    model.eval()
    likelihood.eval()

    test_x = []
    for i in range(n):
        for j in range(n):
            test_x.append([i, j])
    
    test_x = Variable(torch.FloatTensor(np.asarray(test_x))).cuda()
    print("test_x.shape")
    print(test_x.shape)

    test_batch_size = 896
    full_predictions = []
    full_predictions_var = []
    for i in range(0, test_x.shape[0], test_batch_size):
        test_indices = range(test_x.shape[0])[i: i+test_batch_size]
        test_batch_x = test_x[test_indices]

        print(test_batch_x.shape)
               
        # Make binary predictions by warmping the model output through a Bernoulli likelihood
        with gpytorch.beta_features.fast_pred_var():
            predictions = likelihood(model(test_batch_x))
         

            predictions_np = predictions.mean().cpu().data.numpy()
      
            lower, upper = predictions.confidence_region()
            covar_pred = predictions.covar().cpu()

            # print("lower")
            # print(lower)
            # print("upper")
            # print(upper)
            for i in range(n*n):
                print("lower: ", lower[i])
                print("upper: ", upper[i])
                print("covar_pred: ", covar_pred[i])
            full_predictions.append(predictions_np)
           # full_predictions_var.append(predictions_np_var)

    full_predictions = np.asarray(full_predictions)
    full_predictions = np.concatenate(full_predictions, axis=0)

   # full_predictions_var = np.concatenate(np.asarray(full_predictions_var), axis=0)
    print(full_predictions.shape)

    return full_predictions #, full_predictions_var


def plot_result(predictions):

    mask_filenames, train_mask_labels = load_images_from_folder('./masks')

    train_x = []
    train_y = []
    pixel_mask_counts = []
    dict_pixel = {}

    for i in range(len(mask_filenames)):
        img = cv2.imread(mask_filenames[i] ,0)
        mask_label = int(train_mask_labels[i])
        print('has read ', i)
        for j in range(n):
            for k in range(n):
                pixel_position = (j, k)        
                if img[j][k] == 255:
                    if pixel_position in dict_pixel:
                        dict_pixel[pixel_position] += mask_label
                    else:
                        dict_pixel[pixel_position]  = mask_label

    result_gray_img = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            pixel_pos = (i,j)
            if pixel_pos in dict_pixel:
                result_gray_img[i][j] = dict_pixel[pixel_pos]


    result_gray_img_show = result_gray_img.copy()

    result_gray_img_show = result_gray_img_show -result_gray_img_show.min()
    result_gray_img_show = result_gray_img_show/result_gray_img_show.max()
    result_gray_img_show *= 255


    result_gray_img_show = np.array(result_gray_img_show, dtype = np.uint8)
    result_heatmap = cv2.applyColorMap(result_gray_img_show, cv2.COLORMAP_JET)

    # # cv2.imwrite('./weighted_mask/weighted_mask_heatmap.png', result_heatmap)
    cv2.imshow("result_heatmap", result_heatmap)

    org_test_gray_img = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            org_test_gray_img[i][j] = predictions[i*n+j]

           
   
    test_gray_img = org_test_gray_img.copy()
    test_gray_img -= test_gray_img.min()
    test_gray_img /= test_gray_img.max()
    test_gray_img *= 255


    test_gray_img = np.array(test_gray_img, dtype = np.uint8)

  
    test_heatmap = cv2.applyColorMap(test_gray_img, cv2.COLORMAP_JET )

    cv2.imshow("test_heatmap", test_heatmap)
    cv2.waitKey()
    cv2.destroyAllWindows()


    org_img = cv2.imread('original_img_index2_label_0.png')
   


    # final_masked_img = org_img.transpose(2,0,1) * org_test_gray_img 
    # final_masked_img = final_masked_img.transpose(1,2,0)
    # final_masked_img -= final_masked_img.min()
    # final_masked_img /= final_masked_img.max()
    # final_masked_img *= 255
    # final_masked_img = np.array(final_masked_img, dtype = np.uint8)


     # Initialize figiure an axis
    f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
    # # Test points are 100x100 grid of [0,1]x[0,1] with spacing of 1/99

   
    plt.subplot(131),plt.imshow(org_img[:,:,::-1],'gray'),plt.title('Original img')

    plt.subplot(132),plt.imshow(result_heatmap[:,:,::-1],'gray'),plt.title('Summed label training heatmap')


    #plt.subplot(122),plt.imshow(result_gray_img[:,:,::-1],'gray'),plt.title('Summed label training heatmap')
   
    plt.subplot(133),plt.imshow(cv2.cvtColor(test_heatmap, cv2.COLOR_BGR2RGB),'gray'),plt.title('Predicted mask heatmap')
    #plt.subplot(144),plt.imshow(cv2.cvtColor(final_masked_img, cv2.COLOR_BGR2RGB),'gray'),plt.title('Org_img with predicted mask')

    plt.colorbar()
    #plt.set_cmap('Reds')
    plt.set_cmap('jet')
    plt.show()


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

    summed_superpixel_labels = validate(val_loader, model, criterion, eval_img_index)

    for i in range(n):
        for j in range(n):
            pixel_pos = (i,j)
            if pixel_pos in dict_pixel:
                train_x.append([i, j])
                train_y.append(summed_superpixel_labels[i][j])

    
    likelihood = GaussianLikelihood().cuda()

    model = GPRegressionModel(train_x.data, train_y.data, likelihood).cuda()


        # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)


    if mode == 'Train':
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # "Loss" for GPs - the marginal log likelihood
        # n_data refers to the amount of training data
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

       
        train(train_x, train_y, model, optimizer, mll)

    elif mode == 'Eval':
        print("start to test the model")
        predictions = eval_superpixels(model, likelihood)
        plot_result(predictions)
    else:
        raise Exception("No such mode")

    


if __name__== "__main__":
  main()

