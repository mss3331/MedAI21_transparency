#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:36:02 2021

@author: endocv2021@generalizationChallenge
"""

import checkpoints

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import skimage
from skimage import io
from tifffile import imsave
from Detection import detect
import skimage.transform
from main import getModel
from torchvision import models


# class HandSegModel(nn.Module):
#     def __init__(self):
#         super(HandSegModel, self).__init__()
#         self.dl =  models.segmentation.deeplabv3_resnet50(pretrained=False,progress=True,num_classes=2)
#
#     def forward(self, x):
#         y = self.dl(x)['out']
#         return y
def create_predFolder(task_type):
    directoryName = 'Giana21'
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)

    if not os.path.exists(os.path.join(directoryName, task_type)):
        os.mkdir(os.path.join(directoryName, task_type))

    return os.path.join(directoryName, task_type)


def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    # Deeplab Options

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50', 'deeplabv3plus_mobilenet'],
                        help='model name')

    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    return parser


if __name__ == '__main__':
    '''
     You are not allowed to print the images or visualizing the test data according to the rule. 
     We expect all the users to abide by this rule and help us have a fair challenge "EndoCV2021-Generalizability challenge"

     FAQs:
         1) Most of my predictions do not have polyp.
            --> This can be the case as this is a generalisation challenge. The dataset is very different and can produce such results. In general, not all samples 
            have polyp.
        2) What format should I save the predictions.
            --> you can save it in the tif or jpg format. 
        3) Can I visualize the data or copy them in my local computer to see?
            --> No, you are not allowed to do this. This is against challenge rules. No test data can be copied or visualised to get insight. Please treat this as unseen image.!!!
        4) Can I use my own test code?
            --> Yes, but please make sure that you follow the rules. Any visulization or copy of test data is against the challenge rules. We make sure that the 
            competition is fair and results are replicative.
    '''
    # model, device = mymodel()
    # list of models that I can use
    # ************** modify for full experiment *************
    # [SegNet, SegNetGRU, SegNetGRU_Symmetric, SegNetGRU_Symmetric_columns,
    # SegNetGRU_Symmetric_columns_shared_EncDec, SegNetGRU_Symmetric_columns_UltimateShare,
    # SegNetGRU_Symmetric_columns_last2stages, SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec
    # SegNetGRU_Symmetric_columns_last2stages_Notshared_EncDec_smallerH, SegNetGRU_5thStage_only_not_shared,
    # SegNetGRU_4thStage_only_not_shared, SegNetGRU_Symmetric_last2stages_FromEncToDec]
    ########################### Deeplab versions ###################################
    # [Deeplap_resnet50, Deeplap_resnet101, FCN_resnet50, FCN_resnet101, Deeplabv3_GRU_ASPP_resnet50,
    # Deeplabv3_GRU_CombineChannels_resnet50, Deeplabv3_GRU_ASPP_CombineChannels_resnet50, Deeplabv3_LSTM_resnet50]
    model_name = 'Deeplabv3_GRU_CombineChannels_resnet50'
    checkpoint = torch.load('./MedAI21/checkpoints/highest_IOU_' + model_name + '.pt')
    model = getModel(model_name, input_channels=3, number_classes=2)
    model.load_state_dict(checkpoint['state_dict'])

    task_type = 'segmentation'
    # set image folder here!
    # directoryName = create_predFolder(task_type)
    # resize_factor = 3
    # train_img_size = (int(1080 / resize_factor), int(1350 / resize_factor))
    # ----> three test folders [https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/wiki/EndoCV2021-Leaderboard-guide]
    subDirs = ['Polyp', 'Instrument']
    subDirs = ['Polyp']
    for j in range(0, len(subDirs)):

        # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
        imgfolder = '/content/MedAI/test/' + subDirs[j]
        # imgfolder = './testfolders/' + subDirs[j]#************************************Delete this one before uploading!!!!!
        # set folder to save your checkpoints here!
        saveDir = os.path.join(imgfolder + '_pred')

        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        imgfiles = detect_imgs(imgfolder, ext='.jpg')

        # from torchvision import transforms
        # data_transforms = transforms.Compose([
        #     # transforms.RandomResizedCrop(256),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # file = open(saveDir + '/' + "timeElaspsed" + subDirs[j] + '.txt', mode='w')
        timeappend = []

        for imagePath in imgfiles[:]:
            """plt.imshow(img1[:,:,(2,1,0)])
            Grab the name of the file. 
            """
            filename = (imagePath.split('/')[-1]).split('.jpg')[0]
            # filename = (imagePath.split('\\')[-1]).split('.tif')[0]
            print('filename is printing::=====>>', filename)

            # img1 = Image.open(imagePath).convert('RGB').resize((256,256), resample=0)
            # image = data_transforms(img1)
            # perform inference here:
            # images = image.to(device, dtype=torch.float32)

            #
            img = skimage.io.imread(imagePath)
            size = img.shape
            start.record()
            resize_factor = 2  # ************************depending on the size of the images when we had done the training
            # train_img_size = (int(size[0]/resize_factor),int(size[1]/resize_factor))

            train_img_size = (int(531 / resize_factor), int(600 / resize_factor))
            output = detect(model, img, train_img_size)

            #
            # outputs = (images.unsqueeze(0))
            #
            end.record()
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            timeappend.append(start.elapsed_time(end))
            #

            pred = output * 255  # to convert from 0 1 to 0 255
            # pred = (pred).astype('uint8')

            img_mask = skimage.transform.resize(pred, (size[0], size[1]), anti_aliasing=True)
            io.imsave(saveDir + '/' + filename + '.png', (np.round(img_mask / 255) * 255).astype('uint8'))

            # file.write('%s -----> %s \n' %
            #            (filename, start.elapsed_time(end)))

        # file.write('%s -----> %s \n' %
        #            ('average_t', np.mean(timeappend)))
