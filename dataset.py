import torch
import torch.utils.data as data
import numpy
import random
import cv2
import sys
import torch.nn.functional as F
import gzip
import pickle
import numpy as np
import os
import torchvision.transforms as transform
import csv
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from imagenet_lables import *
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.util import img_as_float


class imagenet_localization_dataset(data.Dataset):
    def __init__( self, data_dir, crop, transform = None ):
        import csv
        img_dirs = []
        labels = []
        bboxes_list = []
        img_list_dir = os.path.join( data_dir, 'LOC_val_solution.csv' )
        with open(img_list_dir, 'r') as f:
            for line in f:
                img_name, anno = line.split(',')
                anno = str(anno).split()
                if len(anno) % 5 != 0:
                    continue
                img_dirs.append( os.path.join( data_dir, anno[0], img_name + '.JPEG' ) )
                bboxes = []
                for i in range(len(anno) // 5):
                    x, y, h, w = anno[i*5+1:i*5+5]
                    x, y, h, w = float(x), float(y), float(h), float(w)
                    h -= x
                    w -= y
                    bboxes.append([x, y, h, w])
                bboxes_list.append( bboxes )
                labels.append( anno[0] )

        labels_list = sorted(set(labels))
        dic = {}
        for i in range(len(labels_list)):
            dic[labels_list[i]] = i
        labels = [dic[i] for i in labels]
        self.img_dirs = img_dirs
        self.labels = labels
        self.bboxes_list = bboxes_list
        self.transform = transform
        self.len = len(img_dirs)
        self.crop = crop
       
    
    def __len__( self ):
        return self.len

    def __getitem__( self, index ):
        img = Image.open( self.img_dirs[index] ).convert('RGB')
        bboxes = self.bboxes_list[index]
        label = self.labels[index]

        if self.crop == -1:
            A = None
            for bbox in bboxes:
                x, y, w, h = bbox
                #print(bbox)
                img_w, img_h = img.size
                if img_w < img_h:
                    r = 224 / img_w
                else:
                    r = 224 / img_h
                x, y, w, h = x*r, y*r, w*r, h*r

                img_w = img_w*r
                img_h = img_h*r
                
                if A is None:
                    A = [x, y, w, h]

                B = [0, 0, 224, 224]

                A = bbox_intersection(A, B)

                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

                transform = transforms.Compose([
                            transforms.Resize((int(img_h), int(img_w))),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                            ])
                self.transform = transform
                break

        if self.transform is not None:
            img = self.transform(img)
            print("img.size")
            print(img)
       
        return img, torch.from_numpy(np.array([label])), torch.from_numpy(np.array(A))


def bbox_intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return [0,0,0,0] 
    return [x, y, w, h]


def main():
    # This part is to test the ground truth bounding boxes for ImageNet localization dataset above

    val_data_dir = "/home/lili/Video/GP/examples/network_interpretation_imagenet/data/val"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
                    #transforms.Resize((A[3], A[2])),
                   # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

    val_loader = torch.utils.data.DataLoader(imagenet_localization_dataset(
                    data_dir=val_data_dir,
                    crop = -1,
                    transform=transform),
        batch_size = 1, shuffle=False,
        num_workers = 1, pin_memory=True)
            
    

    count = 0

    eval_img_index = 4
    
    for i, (input, target, gt_bboxes) in enumerate(val_loader):

        count +=1

        if count > eval_img_index:
            break
        
        input_var = torch.autograd.Variable(input, requires_grad=True)
        input_var.requires_grad = True
        target_var = torch.autograd.Variable(target)

        img_show = input[0].numpy().copy()
        img_show = img_show.transpose( 1, 2, 0 )

        img_show -= img_show.min()
        img_show /= img_show.max()
        img_show *= 255
        img_show = img_show.astype(np.uint8)

        label = target[0].numpy()[0]
        
        x, y, w, h = gt_bboxes[0]

        if count == eval_img_index:
            print("input")
            print(input)
            final_img = img_show.copy()
            print("gt_bbox")
            print(gt_bboxes[0])
            cv2.rectangle(final_img,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),2)

            cv2.imshow("org_img_label_{}.png".format(classes_dict[label]), final_img)
            cv2.waitKey(0)
           

            # segments = felzenszwalb(img_as_float(img_show), scale=100, sigma=0.5, min_size=50)
                        
            # print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))

            #cv2.imshow('superpixels', mark_boundaries(img_as_float(img_show), segments))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # compute output
            # output = model(input_var)
            # loss = criterion(output, target_var)

            # pred = output.data.max(1, keepdim=True)[1]
            # label = target[0]
            # print("label ", label)
            # print("pred[0].cpu().numpy() ", pred[0].cpu().numpy()[0])

            # if pred[0].cpu().numpy()[0] == label:
            #     print("correct prediction, index_{} , label_{}".format(count, classes_dict[label]))



if __name__== "__main__":
  main()
