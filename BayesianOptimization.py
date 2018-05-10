import sklearn.gaussian_process as gp
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
from scipy.stats import norm
from scipy.optimize import minimize


from utils import normalize_image
from imagenet_lables import *

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
                summed_superpixel_labels_show = summed_superpixel_labels.copy()[:,:,None]
                print("summed_superpixel_labels_show.shape: ", summed_superpixel_labels_show.shape)
                summed_superpixel_labels_show -= summed_superpixel_labels_show.min()
                summed_superpixel_labels_show /= summed_superpixel_labels_show.max()
                summed_superpixel_labels_show *= 255
                summed_superpixel_labels_show = summed_superpixel_labels_show.astype(np.uint8)
                summed_labels_heatmap = cv2.applyColorMap(summed_superpixel_labels_show, cv2.COLORMAP_JET)
                #cv2.imwrite("summed_superpixel_labels.png", summed_labels_heatmap)
                #cv2.imshow("summed_superpixel_labels", summed_labels_heatmap)

                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                return summed_superpixel_labels, summed_labels_heatmap
            else:
                print("wrong prediction")
                #print("%d samples, the corrrect prediction number: %d "%(len(mask_filenames), correct_pred_count))
            

    return 0

def Jaccard_distance(img1, img2):
    """
    The Jaccard distance between two unmasked area for two input mask images: 1-IOU
    https://en.wikipedia.org/wiki/Jaccard_index
    img1, img2 224x224 gray image

    return: the Jaccard_distance
    """
    assert(img1.shape==img2.shape)

    n = img1.shape[0]

    white_intersect_count = 0
    white_union_count = 0
    for i in range(n):
      for j in range(n):
        if img1[i][j] == 255 or img2[i][j] == 255:
          white_union_count += 1
        if img1[i][j]==img2[i][j]==255:
          white_intersect_count+=1

    IOU = white_intersect_count/white_union_count
    
    Jaccard_distance = 1 - IOU
    print("Jaccard_distance: ", Jaccard_distance)
    return Jaccard_distance


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

return xp, yp


def main():

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

    summed_superpixel_labels, summed_superpixel_labels_heatmap = validate_nueral_network(val_loader, model, criterion, eval_img_index)

    time_duration = time.time()-start_time

    print("time duration is: ", time_duration) 

    

if __name__== "__main__":
  main()


