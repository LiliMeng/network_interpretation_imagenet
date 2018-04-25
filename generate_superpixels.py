
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2

org_img = cv2.imread('orginal_img.png')
segments_org = felzenszwalb(img_as_float(org_img), scale=100, sigma=0.5, min_size=50)
print("org Felzenszwalb number of segments: {}".format(len(np.unique(segments_org))))

img = cv2.imread('frog.png')

segments = felzenszwalb(img_as_float(img), scale=100, sigma=0.5, min_size=50)
            
print("Felzenszwalb number of segments: {}".format(len(np.unique(segments))))

# cv2.imshow('superpixels', mark_boundaries(img_as_float(img), segments))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.subplot(131),plt.imshow(org_img[:,:,::-1],'gray'),plt.title('Org_img', fontsize=60)
plt.subplot(132),plt.imshow(mark_boundaries(img_as_float(org_img[:,:,::-1]), segments_org),'gray'),plt.title('superpixel_org', fontsize=60)
                       
plt.subplot(133),plt.imshow(mark_boundaries(img_as_float(img[:,:,::-1]), segments), 'gray'),plt.title('superpixel_2', fontsize=60)
    
figure = plt.gcf() # get current figure
figure.set_size_inches(80, 30)
                          

plt.savefig('superpixels.png')

