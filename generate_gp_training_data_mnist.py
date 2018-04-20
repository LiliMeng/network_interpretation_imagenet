from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import cv2


from utils import weight_init, save_checkpoint, normalize_image
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import random
from random import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_masked_superpixels', type=int, default=1, metavar='N',
                    help='number of masked superpixels for each image (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_nn = False
prepare_GP_training_data = True


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


def conv( inp_chl, out_chl, ker_size = 3, stride = 1, padding = 1 ):
    return nn.Sequential(
        nn.Conv2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        nn.BatchNorm2d( out_chl ),
        nn.ReLU( True ),
        )

def tconv( inp_chl, out_chl, ker_size = 4, stride = 2, padding = 1 ):
    return nn.Sequential(
        nn.ConvTranspose2d( inp_chl, out_chl, ker_size, stride = stride, padding = padding ),
        nn.BatchNorm2d( out_chl ),
        nn.ReLU( True ),
        )

class Classification_Net( nn.Module ):
    def __init__(self):
        super().__init__()
        self.conv1 = conv( 1, 32 )
        self.conv2 = conv( 32, 32 )
        self.conv3 = conv( 32, 64, stride = 2 )
        self.conv4 = conv( 64, 64 )
        self.conv5 = conv( 64, 128, stride = 2 )
        self.conv6 = nn.Conv2d( 128, 128, 3, padding = 1 )
        self.fc1 = nn.Linear( 128, 10 )

    def forward( self, x ):
        x0 = self.conv2( self.conv1( x  ) )
        x1 = self.conv4( self.conv3( x0 ) )
        x2 = self.conv6( self.conv5( x1 ) )

        f = x2.mean(3).mean(2)
        pred0 = self.fc1( f )

        return x0, x1, x2, pred0

model = Classification_Net()
if args.cuda:
    model.cuda()

optimizer_cls = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)

def train_cls(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer_cls.zero_grad()
        x0, x1, x2, pred0 = model(data)
        output = F.log_softmax(pred0, dim=1)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer_cls.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def eval_cls(load_model=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if load_model == True:
            saved_checkpoint = torch.load('./saved_checkpoints/checkpoint.pth.tar')
            model.load_state_dict( saved_checkpoint['model'] )
        x0, x1, x2, pred0 = model(data)
        output = F.log_softmax(pred0, dim=1)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def eval_superpixel():
    model.eval()
    test_loss = 0
    correct = 0
    saved_checkpoint = torch.load('./saved_checkpoints/mnist/checkpoint.pth.tar')
    model.load_state_dict( saved_checkpoint['model'] )
     
    count =0 
    for data, target in test_loader:
        count +=1
        if count>2:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        org_img = data[0]
      
        org_img = org_img.type(torch.FloatTensor).data
        org_img = org_img.numpy()
        img = org_img.transpose( 1, 2, 0 )
        img -= img.min()
        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
       
        # cv2.imshow('original_img_index{}_label_{}.png'.format(count, target[0].cpu().data.numpy()[0]), img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if count == 2:

            cv2.imwrite('original_img_index{}_label_{}.png'.format(count, target[0].cpu().data.numpy()[0]), img)

            segments = felzenszwalb(img_as_float(img), scale=100, sigma=0.5, min_size=5)
            
            print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))
            colored_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        
            x0, x1, x2, pred0 = model(data)
            output = F.log_softmax(pred0, dim=1)
            pred = output.data.max(1, keepdim=True)[1]
          
            probability_output = F.softmax(pred0, dim=1)
           
            probability_score = probability_output.max(1, keepdim=True)[0]
           
            correct_pred_count=0
            wrong_pred_count =0
            for i in range(1000):
                
                total_num_segments = len(np.unique(segments))
               # num_conse_superpixels = int(0.2*total_num_segments)
               
                num_conse_superpixels = 1
                print("total_num_segments: ", total_num_segments)
                print("num_conse_superpixels: ", num_conse_superpixels)
                firstIndex= randint(1, total_num_segments-num_conse_superpixels)
               

                #random_sampled_list = np.unique(segments)[firstIndex:(firstIndex + num_conse_superpixels)]              
                random_sampled_list= sample(range(np.unique(segments)[0], np.unique(segments)[-1]), num_conse_superpixels)
               
                print("len(random_sampled_list): ", len(random_sampled_list))
                mask = np.zeros(img.shape[:2], dtype= "uint8")
                
                mask.fill(255)
                for (j, segVal) in enumerate(random_sampled_list):
                    mask[segments == segVal] = 0
                    

                masked_img = org_img * mask
                
             
                pic = masked_img


                pic = pic.transpose(1, 2, 0)
                pic -= pic.min()
                pic /= pic.max()
                pic *= 255
                
                pic = np.array(pic, dtype = np.uint8)
               
                
                colored_pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
                mask_heatmap = cv2.applyColorMap(colored_pic, cv2.COLORMAP_JET )
                
                masked_img = normalize_image(masked_img)
                masked_img_batch = masked_img[None, :, :, :]

            
                masked_img_tensor = Variable(torch.from_numpy(masked_img_batch)).cuda()
                x0_mask, x1_mask, x2_mask, pred0_mask = model(masked_img_tensor)
                
                mask_output = F.log_softmax(pred0_mask, dim=1)
                mask_probability_output = F.softmax(pred0_mask, dim=1)
                # print("mask_probability_output")
                # print(mask_probability_output)
                mask_probability_score = mask_probability_output.max(1, keepdim=True)[0]
                # print("mask_probability_score")
                # print(mask_probability_score.cpu().data)
                pred_mask = mask_output.data.max(1, keepdim=True)[1]
                # print("pred_mask[0]", pred_mask[0].cpu().numpy()[0])

                if pred_mask[0].cpu().numpy()[0] == target[0].cpu().data.numpy()[0]:
                    correct_pred_count+=1
                    print("correct_pred_count", correct_pred_count)
               
                    cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 1), mask)
                    cv2.imwrite('./mask_on_img/masked_imgs_{}_pred_{}_{}_{}.png'.format(i, pred_mask[0].cpu().numpy()[0], 1, mask_probability_score.cpu().data.numpy()[0]), pic)
                else:
                    wrong_pred_count +=1
                    print("wrong_pred_count", wrong_pred_count)
                    cv2.imwrite('./masks/mask_{}_{}.png'.format(i, 0), mask)
                    cv2.imwrite('./mask_on_img/masked_imgs_{}_pred_{}_{}_{}.png'.format(i, pred_mask[0].cpu().numpy()[0], 0, mask_probability_score.cpu().data.numpy()[0]), pic)

               
                # plt.subplot(151),plt.imshow(colored_img, 'gray'),plt.title('original_img_label_{}.png'.format(target[0].cpu().data.numpy()[0]))
                # plt.subplot(152),plt.imshow(mark_boundaries(img_as_float(colored_img), segments),'gray'),plt.title('Superpixel')
                # plt.subplot(153),plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 'gray'), plt.title("Mask")
                # plt.subplot(154),plt.imshow(cv2.cvtColor(colored_pic, cv2.COLOR_BGR2RGB),'gray'),plt.title('Org_img with mask gray')
                # plt.subplot(155),plt.imshow(cv2.cvtColor(mask_heatmap, cv2.COLOR_BGR2RGB),'gray'),plt.title('Org_img with mask heatmap pred {}'.format(pred_mask[0].cpu().numpy()))
                # plt.show()
                # plt.close()
        
if train_nn == True:
  for epoch in range(1, 5):
      train_cls(epoch)
      eval_cls(load_model=False)
      save_checkpoint({
                  'epoch': epoch,
                  'model': model.state_dict(),
                  #'optimizer_cls': optimizer_cls.state_dict(),
                  #'optimizer_reg': optimizer_reg.state_dict(),
              }, is_best=False, save_folder="saved_checkpoints" , filename='checkpoint.pth.tar')
elif prepare_GP_training_data == True:
  eval_superpixel()
else:
  raise Exception("Not implemented yet")