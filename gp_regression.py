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
np.set_printoptions(threshold=np.nan)

dataset = 'IMAGENET'

#mode = 'Train'
mode = 'Eval'

if dataset == 'MNIST':
    n = 28
elif dataset == 'CIFAR':
    n = 32
elif dataset == 'IMAGENET':
    n = 224
else:
    raise Exception("This dataset Not implemented yet")

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

def prepare_training_data():
    mask_filenames, train_mask_labels = load_images_from_folder('./masks/')

    train_x = []
    train_y = []
    pixel_mask_counts = []
    dict_pixel = {}

    for i in range(len(mask_filenames)):
        img = cv2.imread(mask_filenames[i] ,0)

        mask_label = int(train_mask_labels[i])
    
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
                train_x.append([i, j])
                train_y.append(dict_pixel[pixel_pos])


    print(result_gray_img)
    result_gray_img_show = result_gray_img.copy()
    print("result_gray_img_show")
    print(result_gray_img_show) 
    result_gray_img_show -= result_gray_img_show.min()
    print("result_gray_img_show.max() ", result_gray_img_show.max())
    result_gray_img_show /= result_gray_img_show.max()
    result_gray_img_show *= 255



    result_gray_img_show = np.array(result_gray_img_show, dtype = np.uint8)
    print("result_gray_img_show")
    print(result_gray_img_show)
    result_heatmap = cv2.applyColorMap(result_gray_img_show, cv2.COLORMAP_JET )

    cv2.imwrite('./weighted_mask/weighted_mask_heatmap.png', result_heatmap)
    cv2.imshow("result_heatmap", result_heatmap)
    cv2.waitKey()
    cv2.destroyAllWindows()



    # for i in range(len(mask_filenames)):
    #     img = cv2.imread(mask_filenames[i] ,0)
    #     mask_label = int(train_mask_labels[i])
    #     for j in range(n):
    #         for k in range(n):
    #             # If the mask make the correct prediction, then these pixels can be masked, each pixel mask has a label 0
    #             if mask_label == 1:
    #                 if img[j][k] == 0:
    #                     train_x.append([j, k])
    #                     train_y.append(1)  
    #             # If the mask make the wrong prediciton, then these pixels cannot be masked, then each pixel mask has a label 1      
    #             elif mask_label == 0:
    #                 if img[j][k] == 0:
    #                     train_x.append([j, k])
    #                     train_y.append(0) 
    #             else:
    #                 raise Exception("No such labels")



    train_x = Variable(torch.FloatTensor(np.asarray(train_x))).cuda()
    train_y = Variable(torch.FloatTensor(np.asarray(train_y))).cuda()
    
    print("train_x.shape: ", train_x.shape)
    print("train_y.shape: ", train_y.shape)

    return train_x, train_y

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
    # Initialize the likelihood and model
    # We use a Gaussian likelihood for regression so we have both a predictive
    # mean and variance for our predictions
    likelihood = GaussianLikelihood().cuda()
    train_x, train_y = prepare_training_data()
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


