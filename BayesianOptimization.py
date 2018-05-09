import sklearn.gaussian_process as gp
import cv2
import numpy as np

def superpixel_IOU(img1, img2):

    n = img1.shape[0]
    white_intersect_count = 0
    for i in range(n):
      for j in range(n):
        if img1[i][j]==img2[i][j]==255:
          white_intersect_count+=1

    white_intersect_count_percent = white_intersect_count*100/(n*n)
    
    print("white_intersect_count_percent: ", white_intersect_count_percent)
    return white_intersect_count_percent

def main():

    img1 = cv2.imread('masks/mask_47_1.png',0)
    img2 = cv2.imread('masks/mask_48_1.png',0)
    superpixel_IOU(img1, img2)

if __name__== "__main__":
  main()


