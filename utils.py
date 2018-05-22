import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil

from torch.autograd import Function
import numpy as np
import cv2

class Binarized( Function ):
    def forward( self, x ):
        output = torch.round( x )
        return output

    def backward( self, output_grad ):
        return output_grad

class Entropy( nn.Module ):
    def __init__( self ):
        super().__init__()
    def forward( self, x ):
        x = nn.Softmax()(x)
        loss = (-x * torch.log(x)).sum(1).mean(0)
        return loss

def cls_zero_grad( m ):
    if hasattr(m, 'cls'):
        m.zero_grad()

def weight_init( m ):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal( m.weight )
        if m.bias is not None:
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

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
        self.avg = self.sum /self.count

def save_checkpoint(state, is_best, save_folder, filename = 'checkpoint.pth.tar'):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    torch.save(state, save_folder + '/' + filename)
    if is_best:
        shutil.copyfile(save_folder + '/' + filename,
                        save_folder + '/' + 'model_best.pth.tar')



class WeightsCheck():
    def __init__(self, model):
        self.params_mean = []
        dtype = torch.FloatTensor
        for param in model.parameters():
            if len(param.size()) == 4 or len(param.size()) == 5:
                self.params_mean.append(float(param.mean().type(dtype)))

    def check(self, model):
        dtype = torch.FloatTensor
        cnt = 0
        for param in model.parameters():
            if len(param.size()) == 4 or len(param.size()) == 5:
                if param.grad is None:
                    print("Warning: param with shape {} has no grad".format(param.size()))
                mean = float(param.mean().type(dtype))
                if mean == self.params_mean[cnt]:
                    print("Warning: param with shape {} has not been updated".format(param.size()))
                self.params_mean[cnt] = mean
                cnt += 1



def normalize_image(image):
    """Convert pixel intensity values from [0, 255] to [0.0, 1.0]."""
    return np.multiply(image.astype(np.float32), 1.0 / 255.0)

def generate_boundingbox(img_index, img, threshold, save_folder):
    """Generate a bounding box for the heatmap"""
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,th1 = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    size = 0
    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        x_, y_, w_, h_ = cv2.boundingRect( contour )
        if w_ * h_ > size:
            x, y, w, h = x_, y_, w_, h_
            size = w * h

    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imwrite(save_folder+"/bbox_{}.png".format(img_index), img)

    return x, y, w, h




def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    IOU = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return IOU