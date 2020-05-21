# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:22:10 2020

@author: Karlo Leskovar
"""
from PIL import Image
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse

import auxiliary #importing the defined functions


parser = argparse.ArgumentParser(description='train.py')


#maknuti -- kod prva dva parsera !!!!!!!
parser.add_argument('data_dir', type= str, action='store', default='./flowers/')

parser.add_argument('--arch', dest='model_type', action='store', default='vgg19', type=str, help='Valid choices are: vgg19, densenet169 and alexnet')

parser.add_argument('--learning_rate', dest='learning_rate', type=float, action='store', default=0.001)

parser.add_argument('--hidden_layer', dest='hidden_size', type=int, action='store', default=2048)

parser.add_argument('--epochs', dest='epochs', type=int, action='store', default=7)

parser.add_argument('--drop_out', dest='drop_out', action='store', type=float, default=0.5)

parser.add_argument('--save_dir', dest='save_dir', action='store', default='./checkpoint_1.pth')

parser.add_argument('--gpu', action='store_true', default= True, 
                    help='Turn GPU mode on or off, default is True.')

parser.add_argument('--batch_size', dest='batch_size', action='store', type=int, default=64, 
                    help='Enter the desired batch size, default is 64.')

args = parser.parse_args()
data_dir = args.data_dir
arch=args.model_type
learning_rate=args.learning_rate
hidden_size=args.hidden_size
epochs=args.epochs
drop_out=args.drop_out
save_dir=args.save_dir
gpu_mode=args.gpu
batch_size=args.batch_size

trainloader, testloader, validloader, train_data = auxiliary.load_images(data_dir, batch_size)

model = auxiliary.Classifier_network(arch, hidden_size, drop_out, learning_rate, gpu_mode)

#print (model.state_dict())

# Recommended to use NLLLoss when using Softmax
criterion = nn.NLLLoss()
# Using Adam optimiser which makes use of momentum to avoid local minima
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

checkpoint = auxiliary.model_training(model, trainloader, validloader, optimizer, criterion, epochs, train_data, gpu_mode)

#print (model.state_dict())

auxiliary.save_model(save_dir, checkpoint)

