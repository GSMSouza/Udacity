
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
from MyModel import MyModel
import functions as fun

arguments = argparse.ArgumentParser()
arguments.add_argument('--input_img', dest="input_img", default='./flowers/test/1/image_06752.jpg', action="store")
arguments.add_argument('--checkpoint',dest="checkpoint", default='model.pth', action='store')
arguments.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
arguments.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
arguments.add_argument('--gpu', action="store_true", dest="gpu",  default=False)

var = arguments.parse_args()

model = MyModel().load(var.checkpoint)
if var.gpu and torch.cuda.is_available():
    model.cuda = var.gpu
else:
    model.cuda = False

with open(var.category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

probabilities = model.predict(fun.process_image(var.input_img), var.top_k)
x = np.array(probabilities[0][0].cpu().detach().numpy(), dtype=float)
y = [cat_to_name[str(model.model.class_to_idx[index])] for index in np.array(probabilities[1][0].cpu().detach().numpy())]

for i in range(len(x)):
    print("Label: {} Probabilitie: {}%".format(y[i], x[i]))


