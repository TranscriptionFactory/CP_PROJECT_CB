# contains model files
import os
import sys
# Torch imports:
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

# Others:
from PIL import Image
import glob
import random
import cv2
import math
import numpy as np
import time
import matplotlib.pyplot as plt

from timm.models.layers import to_2tuple, trunc_normal_

''' Helper for hazy data loader '''
def preparing_training_data(hazefree_images_dir, hazeeffected_images_dir):


    train_data = []
    validation_data = []
    test_data = []
    
    hazy_data = glob.glob(hazeeffected_images_dir + "*.png")

    data_holder = {}

    for h_image in hazy_data:
        h_image = h_image.split("/")[-1] # don't think we need this
        id_ = h_image.split("_")[0] +  ".png" #"_" + h_image.split("_")[1] + ".png"

        if id_ in data_holder.keys():
            data_holder[id_].append(h_image)
        else:
            data_holder[id_] = []
            data_holder[id_].append(h_image)


    train_ids = []
    val_ids = []
    test_ids = []

    num_of_ids = len(data_holder.keys())
    for i in range(num_of_ids):
        if i < num_of_ids*7/10:
            train_ids.append(list(data_holder.keys())[i])
        elif i < num_of_ids*(8.5/10.0):
            val_ids.append(list(data_holder.keys())[i])
        else:
            test_ids.append(list(data_holder.keys())[i])

    for id_ in list(data_holder.keys()):

        if id_ in train_ids:
            for hazy_image in data_holder[id_]:

                train_data.append([hazefree_images_dir + id_, hazeeffected_images_dir + hazy_image])


        elif id_ in val_ids:
            for hazy_image in data_holder[id_]:

                validation_data.append([hazefree_images_dir + id_, hazeeffected_images_dir + hazy_image])
        
        else: #test 
            for hazy_image in data_holder[id_]:

                test_data.append([hazefree_images_dir + id_, hazeeffected_images_dir + hazy_image])


    random.shuffle(train_data)
    random.shuffle(validation_data)
    random.shuffle(test_data)

    return train_data, validation_data, test_data


''' Data loader for training '''
class hazy_data_loader(data.Dataset):

    def __init__(self, hazefree_images_dir, hazeeffected_images_dir, mode='train'):

        self.train_data, self.validation_data, self.test_data = preparing_training_data(hazefree_images_dir, hazeeffected_images_dir) 

        if mode == 'train':
            self.data_dict = self.train_data
            print("Number of Training Images:", len(self.train_data))
        elif mode == 'val':
            self.data_dict = self.validation_data
            print("Number of Validation Images:", len(self.validation_data))
        elif mode == 'test': 
            self.data_dict = self.test_data
            print("Number of Test Images:", len(self.test_data)) 

    def __getitem__(self, index):

        hazefree_image_path, hazy_image_path = self.data_dict[index]

        hazefree_image = Image.open(hazefree_image_path)
        hazy_image = Image.open(hazy_image_path)

        hazefree_image = hazefree_image.resize((128, 128), Image.ANTIALIAS)
        hazy_image = hazy_image.resize((128, 128), Image.ANTIALIAS)

        hazefree_image = (np.asarray(hazefree_image)/255.0) 
        hazy_image = (np.asarray(hazy_image)/255.0) 

        hazefree_image = torch.from_numpy(hazefree_image).float()
        hazy_image = torch.from_numpy(hazy_image).float()

        return hazefree_image.permute(2,0,1), hazy_image.permute(2,0,1)

    def __len__(self):
        return len(self.data_dict)
    
''' Single image dehazing '''
# def image_haze_removel(input_image, directory):

#     hazy_image = (np.asarray(input_image)/255.0)
#     hazy_image = torch.from_numpy(hazy_image).float()
#     hazy_image = hazy_image.permute(2,0,1)
#     hazy_image = hazy_image.cuda().unsqueeze(0)

#     ld_net = LightDehaze_Net().cuda()
#     ld_net.load_state_dict(torch.load(directory))

#     dehaze_image = ld_net(hazy_image)
#     return dehaze_image


# def compute_psnr(img1, img2):
#     img1 = img1.astype(np.float64) / 255.
#     img2 = img2.astype(np.float64) / 255.
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100
#     return 0 * math.log10(255.0 / np.sqrt(mse))
def compute_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def plot_comparison(hazy, test, clear, psnr_vals):
    fig, (ax0, ax1, ax2) = plt.subplots(3,1, figsize = (15, 15))

    ax0.imshow(hazy)

    ax0.set_title('Hazy, PSNR = ' + f'{psnr_vals["hazy"]}')

    ax1.imshow(test)
    ax1.set_title('Dehazed Test, PSNR = ' + f'{psnr_vals["test"]}')

    ax2.imshow(clear)
    ax2.set_title('Clear')

    return [hazy, test, clear, psnr_vals]





