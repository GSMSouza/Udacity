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


class MyModel():
    def __init__(self, model='vgg16', n_neurons = 248, n_neurons2 = 64, 
                 output = 102, learning_rate = 0.001, 
                 dropout = 0.5, cuda = False):
        
        self.dict = {'vgg16': 25088, 'alexnet': 9216, 'densenet161': 2208,
                    'mobilenet_v2':1280}
        
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.name_model = model
        fun = getattr(models, model)
        self.model = fun(pretrained=True)
        if cuda == True and torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False
        
        for param in self.model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([('dropout',nn.Dropout(dropout)),
                                                ('dense1', nn.Linear(self.dict[model], self.n_neurons)),
                                                ('relu1', nn.ReLU()),
                                                ('dense2',nn.Linear(self.n_neurons,n_neurons2)),
                                                ('relu2', nn.ReLU()),
                                                ('dense3',nn.Linear(n_neurons2, output)),
                                                ('output', nn.LogSoftmax(dim=1))]))
        self.model.classifier = classifier
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), self.learning_rate)
        if self.cuda:
            self.model.cuda()
        
    
    def fit(self, train_loader, valid_loader, epochs = 5):
        
        print_every = 5
        steps = 0
        loss_show=[]
        
        if self.cuda:
            self.model.to('cuda')
        
        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(train_loader):
                steps += 1
                if self.cuda:
                    inputs,labels = inputs.to('cuda'), labels.to('cuda')
        
                self.optimizer.zero_grad()
        
       
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
                running_loss += loss.item()
                if steps % print_every == 0:
                    self.model.eval()
                    vlost = 0
                    accuracy=0
                  
                    for ii, (inputs2,labels2) in enumerate(valid_loader):
                        self.optimizer.zero_grad()
                    
                        if self.cuda:
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                            self.model.to('cuda')
                        
                        with torch.no_grad():    
                            outputs = self.model.forward(inputs2)
                            vlost = self.criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                    vlost = vlost / len(valid_loader)
                    accuracy = accuracy /len(valid_loader)
            
                    
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every),
                          "Validation Lost {:.4f}".format(vlost),
                          "Accuracy: {:.4f}".format(accuracy))
           
            
                    running_loss = 0
                
    def evaluation(self, test_loader):
        correct = 0
        total = 0
        if self.cuda:
            self.model.to('cuda')
        with torch.no_grad():
            for line in test_loader:
                images, labels = line
                if self.cuda:
                    images, labels = images.to('cuda'), labels.to('cuda')
                outputs = self.model(images)
                err, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print("Acuracy: {}%".format(100 * correct / total))
        return (correct / total)
          
    def predict(self, image, topk=5):
        image = image.unsqueeze_(0)
        image = image.float()
        
        if self.cuda:
            self.model.to('cuda')
            out = self.model.forward(image.to('cuda'))    
        else:
            self.model.to('cpu')
            out = self.model.forward(image)       
        return F.softmax(out, dim=1).topk(topk)
       
    def save(self, dic, name_file = 'model.pth'):
        self.model.to('cpu')
        idx = {}
        for key in dic.keys():
            idx[dic[key]] = key
            
        torch.save({'model' :self.name_model,
                    'idx': idx,
                    'n_neurons':self.n_neurons,
                    'cuda': self.cuda,
                    'learning_rate': self.learning_rate,
                    'dropout': self.dropout,
                    'dict': self.dict,
                    'state_dict':self.model.state_dict,
                    'optimizer':self.optimizer.state_dict,
                    'classifier': self.model.classifier},
                    name_file)
        
    def load(self, file = 'model.pth'):
        dic = torch.load(file)
        obj = MyModel(dic['model'], n_neurons = dic['n_neurons'],
                      learning_rate = dic['learning_rate'], dropout = dic['dropout'], 
                      cuda = dic['cuda'])
        obj.model.classifier = dic['classifier']
        obj.model.load_state_dict = dic['state_dict']
        obj.model.class_to_idx = dic['idx']
        obj.model.eval()
        return obj

