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

mask_filenames, train_mask_labels = load_images_from_folder('./masks')

n = 32

train_x = []
train_y = []
pixel_mask_counts = []
dict_pixel = {}

for i in range(len(mask_filenames)):
    img = cv2.imread(mask_filenames[i] ,0)
    mask_label = int(train_mask_labels[i])
   
    for i in range(n):
        for j in range(n):
            pixel_position = (i, j)
            if img[i][j] == 0:
                if pixel_position in dict_pixel:
                    dict_pixel[pixel_position] += mask_label
                else:
                    dict_pixel[pixel_position]  = mask_label
          
     
print("dict_pixel")
print(dict_pixel)
position_data = []
label_data = []

for key, value in dict_pixel.items():
    position_data.append(list(key))
    label_data.append(value)

position_data = np.asarray(position_data)
label_data = np.asarray(label_data)


pixel_x, pixel_y = position_data.T
label_data = label_data.T


result_gray_img = np.zeros((32,32,1), np.int8)
for i in range(n):
    for j in range(n):
        pixel_pos = (i,j)
        if pixel_pos in dict_pixel:
            result_gray_img[i][j] = dict_pixel[pixel_pos]


result_gray_img -= result_gray_img.min()
result_gray_img = result_gray_img/ result_gray_img.max()
result_gray_img *= 255

cv2.imwrite('./weighted_mask/weighted_mask.png', result_gray_img)


#cv2.imshow("result_img", result_gray_img)       
result_gray_img = np.array(result_gray_img, dtype = np.uint8)
result_heatmap = cv2.applyColorMap(result_gray_img, cv2.COLORMAP_JET )

cv2.imwrite('./weighted_mask/weighted_mask_heatmap.png', result_heatmap)
# cv2.imshow("result_heatmap", result_heatmap)
# cv2.waitKey()
# cv2.destroyAllWindows()

for i in range(len(mask_filenames)):
    img = cv2.imread(mask_filenames[i] ,0)
    mask_label = int(train_mask_labels[i])
    for i in range(n):
        for j in range(n):
            # If the mask make the correct prediction, then these pixels can be masked, each pixel mask has a label 0
            if mask_label == 1:
                if img[i][j] == 0:
                    train_x.append([i, j])
                    train_y.append(0)  
            # If the mask make the wrong prediciton, then these pixels cannot be masked, then each pixel mask has a label 1      
            elif mask_label == 0:
                if img[i][j] == 0:
                    train_x.append([i, j])
                    train_y.append(1) 
            else:
                raise Exception("No such labels")



train_x = Variable(torch.FloatTensor(np.asarray(train_x))).cuda()
train_y = Variable(torch.FloatTensor(np.asarray(train_y))).cuda()


print(train_x.shape)
print(train_y.shape)

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
                                                    grid_bounds=[(0, 32), (0, 32)])
        # Register the log lengthscale as a trainable parametre
        self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-5,6))
        
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        return GaussianRandomVariable(mean_x, covar_x)

# Initialize the likelihood and model
# We use a Gaussian likelihood for regression so we have both a predictive
# mean and variance for our predictions
likelihood = GaussianLikelihood().cuda()
model = GPRegressionModel(train_x.data, train_y.data, likelihood).cuda()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
    training_iterations = 30
    for i in range(training_iterations):
        # Zero out gradients from backprop
        optimizer.zero_grad()
        # Get predicitve mean and variance
        output = model(train_x)
        # Calculate loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.data[0]))
        optimizer.step()

# Set model and likelihood into evaluation mode
model.eval()
likelihood.eval()


test_x = []
for i in range(n):
    for j in range(n):
        test_x.append([i, j])
      

test_x = Variable(torch.FloatTensor(np.asarray(test_x))).cuda()

print("test_x.shape")
print(test_x.shape)

# Calculate mean and predictive variance
observed_pred = likelihood(model(test_x))
# Labels are predictive means
predictions = observed_pred.mean().view(n, n).data.numpy()


print("predictions")
print(predictions)

# org_test_gray_img = np.zeros((32,32))

# for i in range(n):
#     for j in range(n):
#         print("predictions.mean().cpu().data.numpy()[i*n+j]")
#         print(predictions.mean().cpu().data.numpy()[i*n+j])
#         org_test_gray_img[i][j] = predictions.mean().cpu().data.numpy()[i*n+j]
       

# test_gray_img = org_test_gray_img
# test_gray_img -= test_gray_img.min()
# test_gray_img = test_gray_img/ test_gray_img.max()
# test_gray_img *= 255


# test_gray_img = np.array(test_gray_img, dtype = np.uint8)

# cv2.imwrite('./weighted_mask/pred_mask.png', test_gray_img)

# test_heatmap = cv2.applyColorMap(test_gray_img, cv2.COLORMAP_JET )
# cv2.imwrite('./weighted_mask/pred_mask_heatmap.png', test_heatmap)



# org_img = cv2.imread('original_img_index2_label_8.png')


# # final_masked_img = org_img * org_test_gray_img 
# # #final_masked_img = final_masked_img.transpose(1, 2, 0)
# # final_masked_img -= final_masked_img.min()
# # final_masked_img /= final_masked_img.max()
# # final_masked_img *= 255
# # final_masked_img = np.array(final_masked_img, dtype = np.uint8)
# # final_masked_img_color = cv2.cvtColor(final_masked_img, cv2.COLOR_GRAY2RGB)



# plt.subplot(131),plt.imshow(org_img,'gray'),plt.title('Original img')
# plt.subplot(132),plt.imshow(cv2.cvtColor(result_heatmap, cv2.COLOR_BGR2RGB),'jet'),plt.title('Summed label training heatmap')
# plt.subplot(133),plt.imshow(cv2.cvtColor(test_heatmap, cv2.COLOR_BGR2RGB),'gray'),plt.title('Predicted mask heatmap')
# #plt.subplot(144),plt.imshow(cv2.cvtColor(final_masked_img_color, cv2.COLOR_BGR2RGB),'gray'),plt.title('Org_img with predicted mask')

plt.show()