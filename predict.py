# -*- coding: utf-8 -*-
"""
Created on Mon May 19 07:58:44 2020

@author: Karlo Leskovar
"""

import numpy as np
import argparse
import json
from torchvision import datasets, transforms, models
import auxiliary #importing the defined functions

parser = argparse.ArgumentParser(description='predict.py')


#maknuti -- kod prvog parsera!!!!!
parser.add_argument('--image_path', type=str, default='./flowers/test/25/image_06611.jpg',
                    help='Enter the path to the desired image, default leads to a Grape hyacinth.')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = 'checkpoint_1.pth',
                    help='Enter location where the checkpoint is saved')

parser.add_argument('--top_k', dest='top_k', action='store', type=int, default=5,
                    help='Enter the number of top most likely classes, default is 5.')

parser.add_argument('--arch', action='store',
                    dest='pretrained_model', default='vgg19',
                    help='Enter pretrained model to use, default is VGG-19.')

parser.add_argument('--categories', dest='categories', action='store', default='cat_to_name.json')

parser.add_argument('--gpu', dest='gpu', action='store', default=True, 
                    help='Turn GPU mode on or off, default is on.')

args=parser.parse_args()
input_image=args.image_path
topk=args.top_k
gpu_mode=args.gpu
save_dir = args.save_directory
categories=args.categories

trainloader, testloader, validloader, train_data = auxiliary.load_images()

# Establish model template
pre_tr_model = args.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)

model = auxiliary.model_loader(model, save_dir, gpu_mode)

with open(categories, 'r') as json_file:
    cat_to_name = json.load(json_file)
    
probs, classes = auxiliary.predict(input_image, model, cat_to_name, topk)

if categories != '':
   with open(categories, 'r') as f:
       cat_to_name = json.load(f)
   classes = [cat_to_name[x] for x in classes]    

for i,j in zip(probs, classes):
    print ('The selected flower is most likely a {}, with a probability of {:.3f} %.'.format(j.capitalize(), i*100))