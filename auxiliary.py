# -*- coding: utf-8 -*-
"""
Created on Mon May 18 07:49:31 2020

@author: Karlo Leskovar
"""

from PIL import Image
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models



def load_images(data_dir='./flowers', batch_size=64):
    """
    Function which takes data from flowers folder and does all the required tranformations.
    Returns: trainloader, testloader, validloader, train_data, valid_data, test_data
    """
    
    #image manipulation
    image_rotation = 37
    image_resize = 480
    image_reduction = 255
    center_crop = 224
    
    
    #normalization data
    means_norm = [0.485, 0.456, 0.406]
    stds_norm = [0.229, 0.224, 0.225]
    
    #loader controls
    shuffle = True
    batch_size = batch_size
    
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    train_transform = transforms.Compose([transforms.RandomRotation(image_rotation),
                                         transforms.RandomResizedCrop(image_resize),
                                         transforms.RandomVerticalFlip(),
                                         transforms.Resize(image_reduction), 
                                         transforms.CenterCrop(center_crop),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means_norm,stds_norm)])

    test_transform = transforms.Compose([transforms.Resize(image_reduction),
                                         transforms.CenterCrop(center_crop),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means_norm,stds_norm)])

    valid_transform = transforms.Compose([transforms.Resize(image_reduction),
                                         transforms.CenterCrop(center_crop),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means_norm,stds_norm)])

    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
    


    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = shuffle)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size)

    
    return trainloader, testloader, validloader, train_data


model_name = {'vgg19':25088,
              'densenet169':1664,
              'alexnet':9216}



def Classifier_network(architecture='vgg19', hidden_size=2048, drop_out=0.3, learning_rate=0.001, gpu_mode=True):
    """
    We can choose the architecture of our model (vgg19, densenet169 or alexnet) the size of the hidden layer and the dropout.
    Our model has 1 hidden layer with ReLU activation and dropout.
    Default dopout is set to 0.3 and the default learnrate to 0.001.
    Also we need to select the gpu or cpu as the procesor. 

    """  
    if architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif architecture == 'densenet169':
        model = models.densenet121(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a valid model.You can choose either vgg19, densenet169 or alexnet.".format(architecture))
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model_name[architecture], hidden_size)),
        ('drop_out', nn.Dropout(p=drop_out)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_size, 102)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))
    
    model.classifier = classifier

    
    """cheks if a gpu with cude cores is available and if the selected the gpu 
    as the procesor"""
    if torch.cuda.is_available() and gpu_mode == True:
        model.cuda()
    
    return model

def model_validation(model, validloader, criterion, gpu_mode):
    """
    First a validation function is created, to use in the training function, 
    and later to be used to calculate the accuracy on the test dataset. 
    """
    if gpu_mode == True:
    # change model to work with cuda
        model.to('cuda')
    else:
        pass
    
    test_loss = 0
    accuracy = 0

    #loop through images in validation/test data
    for images, labels in validloader:
         
        if gpu_mode == True:
        # Change images and labels to work with cuda
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass       
        #calculate log-probabilities
        logps = model(images)
        test_loss += criterion(logps, labels).item()
        #calculate probabilities
        ps = torch.exp(logps)
        #check the prediction of the model   
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        #calculate validation/test data accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    #the function returns validation accuracy and validation loss    
    return test_loss, accuracy



def model_training(model, trainloader, validloader, optimizer, criterion, 
                   epochs, train_data, gpu_mode):
    
    """
    A training function, which outputs realevant training data, epoch number, 
    training loss, validation loss and validation accuracy.
    """
    if gpu_mode == True:
    # change to cuda
        model.to('cuda')
    else:
        pass
    
    
    print_every = 15
    steps = 0
    print ("Traning started...")
    for e in range(epochs):
        
        running_loss = 0
        for ii, (images, labels) in enumerate(trainloader):
            steps += 1

            if gpu_mode == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = model_validation(model, validloader, criterion, gpu_mode)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Valid Accuracy: {:.3f} %".format(accuracy/len(validloader)*100))

                running_loss = 0

                model.train()
        print ("{}. epoch done..".format(e+1))
    

    print ("Training done.")
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
            'classifier': model.classifier,
            'epoch': e+1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx
            }
    print('Checkpoint created.')
    return checkpoint


def save_model(save_dir, checkpoint):
    """
    Saving: feature weights, new classifier, index-to-class mapping, 
    optimiser state, and No. of epochs
    """
    
    print ("The model has been saved.")
    
    return torch.save(checkpoint, save_dir)
    



def model_loader(model, save_dir, gpu_mode): 
    """
    The function loads a checkpoint from the model.
    """
    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
        
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

    


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        Returns:
    '''
    
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    image_tensor = transform(image).float()
    
    return image_tensor


#need to pass 5 to topk --> prediciton.topk(5, dim = 1)
def predict(image_path, loaded_model, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    loaded_model.to('cpu')
    loaded_model.eval()
    class_to_idx = loaded_model.class_to_idx
    idx_to_class = {class_to_idx[i]: i for i in class_to_idx}
    #using the previously defined function to load an image
    processed_image = process_image(image_path)

    #unsqueeze the image
    processed_image.unsqueeze_(0)
    with torch.no_grad():
        output = loaded_model(processed_image)
        preds=torch.exp(output).topk(topk)
        
    probs = preds[0][0].cpu().data.numpy().tolist()
    classes = preds[1][0].cpu().data.numpy()
    classes = [idx_to_class[i] for i in classes]
    
    return probs, classes


