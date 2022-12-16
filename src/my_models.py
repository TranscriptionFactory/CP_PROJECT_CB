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
import pandas as pd

from timm.models.layers import to_2tuple, trunc_normal_
torch.backends.cuda.matmul.allow_tf32 = False
from helpers import preparing_training_data, hazy_data_loader, compute_psnr, plot_comparison
from mutual_information import *

def getHelp():
    response = 'Models available: \n LightDehazeNet \n LightDehazeNet_KL \n LighDehazeNet_GL \n LightDehazeNet_MI \n LightDehaze_Net_Attn, LightDehazeNet_Attn_Conv, LightDehazeNet_Attn_Conv_Big'
    print(response)
    return response, ['LightDehazeNet', 'LightDehazeNet_KL', 'LighDehazeNet_GL', 'LightDehazeNet_MI', 'LightDehazeNet_Attn',
                     'LightDehazeNet_Attn_Conv', 'LightDehazeNet_Attn_Conv_Big']


########################################################
        # Original model 
########################################################


class LightDehazeNet:
    def __init__(self, dirpath):
        
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        
        self.directory = dirpath
#         self.ld_net = self.LightDehaze_Net().cuda()
        
    class LightDehaze_Net(nn.Module):

        def __init__(self):
            super(LightDehazeNet.LightDehaze_Net, self).__init__()

            # LightDehazeNet Architecture 
            self.relu = nn.ReLU(inplace=True)

            self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True) 
            self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True) 
            self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True) 
            self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True) 
            self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True) 
            self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True) 
            self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
            self.e_conv_layer8 = nn.Conv2d(56,3,3,1,1,bias=True)

        def forward(self, img):
            pipeline = []
            pipeline.append(img)

            conv_layer1 = self.relu(self.e_conv_layer1(img))
            conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
            conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))

            # concatenating conv1 and conv3
            concat_layer1 = torch.cat((conv_layer1,conv_layer3), 1)

            conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
            conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
            conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))

            # concatenating conv4 and conv6
            concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)

            conv_layer7= self.relu(self.e_conv_layer7(concat_layer2))

            # concatenating conv2, conv5, and conv7
            concat_layer3 = torch.cat((conv_layer2,conv_layer5,conv_layer7),1)

            conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))


            dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1) 
            #J(x) = clean_image, k(x) = x8, I(x) = x, b = 1

            return dehaze_image  
    
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    ''' Model train ''' 
    def train(self, train_og, train_hz, n_epochs,learning_rate ):

        ld_net = self.LightDehaze_Net().cuda()
        ld_net.apply(self.weights_init)

        training_data = hazy_data_loader(train_og, train_hz)
        validation_data = hazy_data_loader(train_og, train_hz, mode = 'val')


        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        
        criterion = nn.MSELoss().cuda()

        optimizer = torch.optim.Adam(ld_net.parameters(), lr=float(learning_rate), weight_decay=0.0001)

        ld_net.train()

        train_loss = []
        val_loss = []
        
        per_epoch_train = []
        per_epoch_val = []
        
        
        num_of_epochs = int(n_epochs)
        for epoch in range(num_of_epochs):
            
            avg_train_loss = []
            avg_val_loss = []
            
            for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):

                hazefree_image = hazefree_image.cuda()/255.0
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)

                loss = criterion(dehaze_image, hazefree_image)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(ld_net.parameters(),0.1)
                optimizer.step()

                train_loss.append(loss.item())
                
                avg_train_loss.append(loss.item())

            # Validation Stage
            for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):

                hazefree_image = hazefree_image.cuda()/255.0
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)
                
                loss = criterion(dehaze_image, hazefree_image)
                
                val_loss.append(loss.item())
                
                avg_val_loss.append(loss.item())
            
            
            per_epoch_train.append(np.array(avg_train_loss).mean())
            per_epoch_val.append(np.array(avg_val_loss).mean())
            
            
            pd.DataFrame(train_loss).to_csv(self.directory + "/train_loss_all.csv")
            pd.DataFrame(val_loss).to_csv(self.directory + "/val_loss_all.csv")
            pd.DataFrame(per_epoch_train).to_csv(self.directory + "/per_epoch_train.csv")
            pd.DataFrame(per_epoch_val).to_csv(self.directory + "/per_epoch_val.csv")
            torch.save(ld_net.state_dict(), self.directory + "/" + str(epoch) + "_trained_LDNet.pth")               

#             torch.save(ld_net.state_dict(), self.directory + "trained_LDNet.pth") 
            
    ''' Single image dehazing '''
    def image_haze_removel(self, input_image):

        hazy_image = (np.asarray(input_image)/255.0)
        hazy_image = torch.from_numpy(hazy_image).float()
        hazy_image = hazy_image.permute(2,0,1)
        hazy_image = hazy_image.cuda().unsqueeze(0)

        ld_net = self.LightDehaze_Net().cuda()
        ld_net.load_state_dict(torch.load(self.directory + "trained_LDNet.pth"))

        dehaze_image = ld_net(hazy_image)
        return dehaze_image
    
    def compare_image(self, validation_data, img_dirs, plot = False):

        hazy = np.asarray(Image.open(validation_data[1]))
        clear = np.asarray(Image.open(validation_data[0]))

        # run image through pipeline
        test_tensor = self.image_haze_removel(Image.open(validation_data[1]))

        test_img = test_tensor[0, :, : :].permute(1, 2, 0)
        test_img_plt = test_img.cpu().data.numpy()

        psnr_vals = {'hazy' : compute_psnr(clear, hazy), 'test' : compute_psnr(clear, test_img_plt)}

        if plot:
            return plot_comparison(hazy, test_img_plt, clear, psnr_vals)
        else:
            return psnr_vals
    

        
########################################################
        # KL-Divergence Loss
########################################################

class LightDehazeNet_KL(LightDehazeNet):
    def __init__(self, dirpath):
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        
        self.directory = dirpath
#         self.ld_net = self.LightDehaze_Net().cuda()
        
    ''' only thing changed in this class is training '''
    def train(self, train_og, train_hz, n_epochs,learning_rate ):

        ld_net = self.LightDehaze_Net().cuda()
        ld_net.apply(self.weights_init)
        
        training_data = hazy_data_loader(train_og, train_hz)
        validation_data = hazy_data_loader(train_og, train_hz, mode = 'val')

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        
        criterion = nn.KLDivLoss(reduction = "batchmean", log_target = True)
        optimizer = torch.optim.Adam(ld_net.parameters(), lr=float(learning_rate), weight_decay=0.0001)

        ld_net.train()

        train_loss = []
        val_loss = []
        
        per_epoch_train = []
        per_epoch_val = []
        
        
        num_of_epochs = int(n_epochs)
        for epoch in range(num_of_epochs):
            
            avg_train_loss = []
            avg_val_loss = []
            
            for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):

                hazefree_image = hazefree_image.cuda()/255.0
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)

                loss = criterion(F.log_softmax(dehaze_image, dim = 1), F.log_softmax(hazefree_image, dim = 1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(ld_net.parameters(),0.1)
                optimizer.step()

                train_loss.append(loss.item())
                
                avg_train_loss.append(loss.item())

            # Validation Stage
            for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):

                hazefree_image = hazefree_image.cuda()/255.0
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)
                
                loss = criterion(F.log_softmax(dehaze_image, dim = 1), F.log_softmax(hazefree_image, dim = 1))
                
                val_loss.append(loss.item())
                
                avg_val_loss.append(loss.item())
            
            
            per_epoch_train.append(np.array(avg_train_loss).mean())
            per_epoch_val.append(np.array(avg_val_loss).mean())
            
            
            pd.DataFrame(train_loss).to_csv(self.directory + "/train_loss_all.csv")
            pd.DataFrame(val_loss).to_csv(self.directory + "/val_loss_all.csv")
            pd.DataFrame(per_epoch_train).to_csv(self.directory + "/per_epoch_train.csv")
            pd.DataFrame(per_epoch_val).to_csv(self.directory + "/per_epoch_val.csv")  

            torch.save(ld_net.state_dict(), self.directory + "/" + str(epoch) + "_trained_LDNet.pth")               

#             torch.save(ld_net.state_dict(), self.directory + "trained_LDNet.pth") 

########################################################
        # Gaussian Loss Function
########################################################
        
class LightDehazeNet_GL(LightDehazeNet):
    def __init__(self, dirpath):
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        
        self.directory = dirpath
#         self.ld_net = self.LightDehaze_Net().cuda()
        
    ''' only thing changed in this class is training '''
    def train(self, train_og, train_hz, n_epochs,learning_rate ):

        ld_net = self.LightDehaze_Net().cuda()
        ld_net.apply(self.weights_init)
        
        training_data = hazy_data_loader(train_og, train_hz)
        validation_data = hazy_data_loader(train_og, train_hz, mode = 'val')

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        
        criterion = nn.GaussianNLLLoss()

        optimizer = torch.optim.Adam(ld_net.parameters(), lr=float(learning_rate), weight_decay=0.0001)

        ld_net.train()

        train_loss = []
        val_loss = []
        
        per_epoch_train = []
        per_epoch_val = []
        
        
        num_of_epochs = int(n_epochs)
        for epoch in range(num_of_epochs):
            
            avg_train_loss = []
            avg_val_loss = []
            
            for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):

                hazefree_image = hazefree_image.cuda()/255.0
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)

                loss = criterion(dehaze_image.cuda(), hazefree_image, torch.ones(dehaze_image.shape).cuda())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(ld_net.parameters(),0.1)
                optimizer.step()

                train_loss.append(loss.item())
                
                avg_train_loss.append(loss.item())

            # Validation Stage
            for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):

                hazefree_image = hazefree_image.cuda()/255.0
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)
                
                loss = criterion(dehaze_image.cuda(), hazefree_image, torch.ones(dehaze_image.shape).cuda())
                
                val_loss.append(loss.item())
                
                avg_val_loss.append(loss.item())
            
            
            per_epoch_train.append(np.array(avg_train_loss).mean())
            per_epoch_val.append(np.array(avg_val_loss).mean())
            
            
            pd.DataFrame(train_loss).to_csv(self.directory + "/train_loss_all.csv")
            pd.DataFrame(val_loss).to_csv(self.directory + "/val_loss_all.csv")
            pd.DataFrame(per_epoch_train).to_csv(self.directory + "/per_epoch_train.csv")
            pd.DataFrame(per_epoch_val).to_csv(self.directory + "/per_epoch_val.csv")  
            torch.save(ld_net.state_dict(), self.directory + "/" + str(epoch) + "_trained_LDNet.pth")               

#             torch.save(ld_net.state_dict(), self.directory + "trained_LDNet.pth")             
########################################################
        # Mutual Information Loss function
########################################################
                 
class LightDehazeNet_MI(LightDehazeNet):
    def __init__(self, dirpath):
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
                
        
        self.directory = dirpath
#         self.ld_net = self.LightDehaze_Net().cuda()
        
    ''' only thing changed in this class is training '''
    def train(self, train_og, train_hz, n_epochs,learning_rate ):

        ld_net = self.LightDehaze_Net().cuda()
        ld_net.apply(self.weights_init)
        
        training_data = hazy_data_loader(train_og, train_hz)
        validation_data = hazy_data_loader(train_og, train_hz, mode = 'val')

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        
        criterion = MutualInformation(num_bins=256, sigma=0.1, normalize=True).to('cuda')

        optimizer = torch.optim.Adam(ld_net.parameters(), lr=float(learning_rate), weight_decay=0.0001)

        ld_net.train()

        train_loss = []
        val_loss = []
        
        per_epoch_train = []
        per_epoch_val = []
        
        
        num_of_epochs = int(n_epochs)
        for epoch in range(num_of_epochs):
            
            avg_train_loss = []
            avg_val_loss = []
            
            for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):

                hazefree_image = hazefree_image.cuda()/255.0
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)

                loss = -criterion(dehaze_image.to('cuda'), hazefree_image.to('cuda'))

                optimizer.zero_grad()
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm(ld_net.parameters(),0.1)
                optimizer.step()

                train_loss.append(loss.sum().cpu().data.numpy())
                
                avg_train_loss.append(loss.sum().cpu().data.numpy())

            # Validation Stage
            for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):

                hazefree_image = hazefree_image.cuda()/255.0
                hazy_image = hazy_image.cuda()

                dehaze_image = ld_net(hazy_image)
                
                loss = -criterion(dehaze_image.to('cuda'), hazefree_image.to('cuda'))
                
                val_loss.append(loss.sum().cpu().data.numpy())
                
                avg_val_loss.append(loss.sum().cpu().data.numpy())
            
            
            per_epoch_train.append(np.array(avg_train_loss).mean())
            per_epoch_val.append(np.array(avg_val_loss).mean())
            
            
            pd.DataFrame(train_loss).to_csv(self.directory + "/train_loss_all.csv")
            pd.DataFrame(val_loss).to_csv(self.directory + "/val_loss_all.csv")
            pd.DataFrame(per_epoch_train).to_csv(self.directory + "/per_epoch_train.csv")
            pd.DataFrame(per_epoch_val).to_csv(self.directory + "/per_epoch_val.csv")   
            torch.save(ld_net.state_dict(), self.directory + "/" + str(epoch) + "_trained_LDNet.pth")               

#             torch.save(ld_net.state_dict(), self.directory + "/" + epoch + "_trained_LDNet.pth")               
########################################################
        # Attention layer
########################################################
cur_folder = 'src'

sys.path.append(str(os.getcwd()[:-len(cur_folder)]) +
'DehazeFormer/')
from models.dehazeformer import *               
class LightDehazeNet_Attn(LightDehazeNet):

    def __init__(self, dirpath):
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        
        
        self.directory = dirpath
#         self.ld_net = self.LightDehaze_Net().cuda()
        
    ''' Different architecture, same training ''' 
    class LightDehaze_Net(LightDehazeNet.LightDehaze_Net):

        def __init__(self):
            super(LightDehazeNet_Attn.LightDehaze_Net, self).__init__()

            # LightDehazeNet Architecture 
            self.relu = nn.ReLU(inplace=True)

            self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True) 
            self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True) 
            self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True) 
            self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True)
            self.attn = Attention(16, 16, 16, 8, 8, use_attn=True)
            self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True) 
            self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True) 
            self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
            self.e_conv_layer8 = nn.Conv2d(56,3,3,1,1,bias=True)

        def forward(self, img):
            pipeline = []
            pipeline.append(img)

            conv_layer1 = self.relu(self.e_conv_layer1(img))
            conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
            conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))

            # concatenating conv1 and conv3
            concat_layer1 = torch.cat((conv_layer1,conv_layer3), 1)

            conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
            attn_layer = self.attn(conv_layer4)
            conv_layer5 = self.relu(self.e_conv_layer5(attn_layer))
    #         conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
            conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))

            # concatenating conv4 and conv6
            concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)

            conv_layer7= self.relu(self.e_conv_layer7(concat_layer2))

            # concatenating conv2, conv5, and conv7
            concat_layer3 = torch.cat((conv_layer2,conv_layer5,conv_layer7),1)

            conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))


            dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1) 
            #J(x) = clean_image, k(x) = x8, I(x) = x, b = 1

            return dehaze_image 

        
########################################################
        # Attention-Conv layer
########################################################
             
class LightDehazeNet_Attn_Conv(LightDehazeNet):

    def __init__(self, dirpath):
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        
        self.directory = dirpath
#         self.ld_net = self.LightDehaze_Net().cuda()
        
    ''' Different architecture, same training '''         
    class LightDehaze_Net(nn.Module):

        def __init__(self):
            super(LightDehazeNet_Attn_Conv.LightDehaze_Net, self).__init__()

            # LightDehazeNet Architecture 
            self.relu = nn.ReLU(inplace=True)

            self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True) 
            self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True) 
            self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True) 
            self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True)
            self.attn = Attention(16, 16, 16, 8, 8, use_attn=True, conv_type = 'Conv')
            self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True) 
            self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True) 
            self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
            self.e_conv_layer8 = nn.Conv2d(56,3,3,1,1,bias=True)

        def forward(self, img):
            pipeline = []
            pipeline.append(img)

            conv_layer1 = self.relu(self.e_conv_layer1(img))
            conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
            conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))

            # concatenating conv1 and conv3
            concat_layer1 = torch.cat((conv_layer1,conv_layer3), 1)

            conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
            attn_layer = self.attn(conv_layer4)
            conv_layer5 = self.relu(self.e_conv_layer5(attn_layer))
    #         conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
            conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))

            # concatenating conv4 and conv6
            concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)

            conv_layer7= self.relu(self.e_conv_layer7(concat_layer2))

            # concatenating conv2, conv5, and conv7
            concat_layer3 = torch.cat((conv_layer2,conv_layer5,conv_layer7),1)

            conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))


            dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1) 
            #J(x) = clean_image, k(x) = x8, I(x) = x, b = 1

            return dehaze_image 

    
########################################################
        # Big Attention-Conv layer
########################################################
             
class LightDehazeNet_Attn_Conv_Big(LightDehazeNet):

    def __init__(self, dirpath):
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        
        self.directory = dirpath
#         self.ld_net = self.LightDehaze_Net().cuda()
        
    ''' Different architecture, same training '''       
    class LightDehaze_Net(nn.Module):

        def __init__(self):
            super(LightDehazeNet_Attn_Conv_Big.LightDehaze_Net, self).__init__()

            # LightDehazeNet Architecture 
            self.relu = nn.ReLU(inplace=True)

            self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True) 
            self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True) 
            self.attn_layer1 = Attention(8, 8, 8, 8, 8, use_attn=True, conv_type = 'Conv')
            self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True) 

            self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True)
            self.attn_layer2 = Attention(16, 16, 16, 8, 8, use_attn=True, conv_type = 'Conv')
            self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True) 
            self.attn_layer3 = Attention(16, 16, 16, 8, 8, use_attn=True, conv_type = 'Conv')

            self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True) 
            self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
            self.e_conv_layer8 = nn.Conv2d(56,3,3,1,1,bias=True)

        def forward(self, img):
            pipeline = []
            pipeline.append(img)

            conv_layer1 = self.relu(self.e_conv_layer1(img))
            conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))

            attn_layer1 = self.attn_layer1(conv_layer2)

            conv_layer3 = self.relu(self.e_conv_layer3(attn_layer1))

            # concatenating conv1 and conv3
            concat_layer1 = torch.cat((conv_layer1,conv_layer3), 1)

            conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))

            attn_layer2 = self.attn_layer2(conv_layer4)

            conv_layer5 = self.relu(self.e_conv_layer5(attn_layer2))

            attn_layer3 = self.attn_layer3(conv_layer5)

    #         conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
            conv_layer6 = self.relu(self.e_conv_layer6(attn_layer3))

            # concatenating conv4 and conv6
            concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)

            conv_layer7= self.relu(self.e_conv_layer7(concat_layer2))

            # concatenating conv2, conv5, and conv7
            concat_layer3 = torch.cat((conv_layer2,conv_layer5,conv_layer7),1)

            conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))


            dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1) 
            #J(x) = clean_image, k(x) = x8, I(x) = x, b = 1

            return dehaze_image 
