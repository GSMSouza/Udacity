import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
from PIL import Image


def load(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    data_train_transforms  = transforms.Compose([transforms.RandomRotation(50),
                                                 transforms.RandomResizedCrop(224),
                                                 transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_validtest_transforms= transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    ds_train = datasets.ImageFolder(train_dir, transform=data_validtest_transforms)
    ds_test = datasets.ImageFolder(test_dir, transform=data_validtest_transforms)
    ds_valid = datasets.ImageFolder(valid_dir, transform=data_validtest_transforms)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=64, shuffle=True)
    
    return train_loader, test_loader, valid_loader, ds_train.class_to_idx

def process_image(image):
    img = Image.open(image)
    process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = process(img)
    
    return tensor