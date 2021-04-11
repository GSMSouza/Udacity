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
import argparse
from MyModel import MyModel
import functions as fun

arguments= argparse.ArgumentParser(description='train.py')
arguments.add_argument('--data_dir', action="store", default="./flowers/")
arguments.add_argument('--gpu', dest="gpu", action="store_true", default=False)
arguments.add_argument('--save_dir', dest="save", action="store", default="./model.pth")
arguments.add_argument('--learning_rate', dest="learning_rate", action="store", type=float, default=0.001)
arguments.add_argument('--dropout', dest = "dropout", action = "store", type=float, default = 0.5)
arguments.add_argument('--epochs', dest="epochs", action="store",type=int, default=5)
arguments.add_argument('--arch', dest="model", action="store", default="vgg16")
arguments.add_argument('--hidden_units', dest="n_neurons", action="store",type=int, default=244)
arguments.add_argument('--hidden_units2', dest="n_neurons2", action="store",type=int, default=64)
arguments.add_argument('--outputs', dest="outputs", action="store",type=int, default=102)

var = arguments.parse_args()

train, valid, test, idx = fun.load(var.data_dir)

model = MyModel(var.model, n_neurons= var.n_neurons,
                n_neurons2=var.n_neurons2, output=var.outputs,
                learning_rate= var.learning_rate,
                dropout= var.dropout, cuda= var.gpu)

model.fit(train, valid, var.epochs)

model.save(idx, var.save)
