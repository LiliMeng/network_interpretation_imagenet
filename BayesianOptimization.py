import sklearn.gaussian_process as gp
import cv2
import numpy as np

def superpixel_IOU():
    img1 = cv2.imread('masks/mask_47_1.png',0)
    img2 = cv2.imread('masks/mask_48_1.png',0)

    print(img1.shape)
    
    n = img1.shape[0]
    white_intersect_count = 0
    black_intersect_count = 0
    for i in range(n):
      for j in range(n):
        if img1[i][j]==img2[i][j]==255:
          white_intersect_count+=1
        if img1[i][j]==img2[i][j]==0:
          black_intersect_count+=1

    print("white_intersect_count: ", white_intersect_count)
    print("white_intersect_count_percent: ", white_intersect_count*100/(n*n))
    print("black_intersect_count: ", black_intersect_count)
    print("black_intersect_count_percent: ", black_intersect_count*100/(n*n))



def main():
    superpixel_IOU()

if __name__== "__main__":
  main()


