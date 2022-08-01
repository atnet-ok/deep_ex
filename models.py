

import torch
from torch import nn
import timm
from metric import *
from torch.nn.parameter import Parameter
import math
from torch.nn.parameter import Parameter
import math

class Classifer(nn.Module):
    def __init__(self, pre_train_model, class_num, metric,pre_train):
        super(Classifer, self).__init__()
        self.metric = metric

        # for timm
        self.backbone =  timm.create_model(
            pre_train_model, 
            pretrained=pre_train, 
            num_classes= 0
        )
        in_features = self.backbone.num_features

        if metric =='adacos':
            self.head = AdaCos(in_features,class_num)
        elif metric == 'arcface':
            self.head = ArcFace(in_features,class_num)
        elif metric == 'sphereface':
            self.head = SphereFace(in_features,class_num)
        elif metric == 'cosface':
            self.head = CosFace(in_features,class_num)
        else:
            self.head = nn.Linear(in_features, class_num)

    def extract(self, x):
        return self.backbone(x)
    
    def forward(self, x, label = None):
        x = self.backbone(x)
        if self.metric:
            x = self.head(x, label)
        else:
            x = self.head(x)
        return x


class Classifer2(nn.Module):
    def __init__(self, pre_train_model, class_num, metric,pre_train):
        super(Classifer2, self).__init__()
        self.metric = metric

        # for timm
        self.backbone =  timm.create_model(
            pre_train_model, 
            pretrained=pre_train, 
            num_classes= 0
        )
        
        in_features = self.backbone.num_features
        sub_features = 512
        self.subhead = nn.Linear(in_features, sub_features)

        if metric =='adacos':
            self.head = AdaCos(sub_features,class_num)
        elif metric == 'arcface':
            self.head = ArcFace(sub_features,class_num)
        elif metric == 'sphereface':
            self.head = SphereFace(sub_features,class_num)
        elif metric == 'cosface':
            self.head = CosFace(sub_features,class_num)
        else:
            self.head = nn.Linear(sub_features, class_num)

    def extract(self, x):
        x = self.backbone(x)
        x = self.subhead(x)
        return x
    
    def forward(self, x, label = None):
        x = self.backbone(x)
        x = self.subhead(x)

        if self.metric:
            x = self.head(x, label)
        else:
            x = self.head(x)

        return x


def get_model(cfg, pre_train=True):


    #model = Classifer(cfg['pre_train_model'], cfg['class_num'], cfg['metric_learning'],pre_train)
    model = Classifer2(cfg['pre_train_model'], cfg['class_num'], cfg['metric_learning'],pre_train)
    model.to(cfg['device'])
    
    return model

