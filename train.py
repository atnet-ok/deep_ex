
# 必要なものをimport
#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
#https://comp.probspace.com/competitions/religious_art/discussions/anoo-Post43902e99c081b1fbc676
#https://qiita.com/0xfffffff7/items/efbb65521d7708f2db7d
from cProfile import label
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
from torchmetrics import Metric
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import json
import os
import random
import pickle
import timm
from models import get_model
from optimizers import get_optimizer,get_scheduler
from lib import get_transform, save_config, load_config, setup_logger
from pprint import pprint
#from torchsummary import summary
from glob import glob
from sklearn.model_selection import StratifiedKFold
#https://pystyle.info/pytorch-how-to-use-pretrained-model/
#https://qiita.com/illumination-k/items/fa7508127d8942c1284f
#https://qiita.com/yu4u/items/078054dfb5592cbb80cc　深層距離学習
#https://github.com/sthalles/SimCLR  自己教示あり学習

# import albumentations as A
# import albumentations
# #https://nonbiri-tereka.hatenablog.com/entry/2020/08/26/084816
# import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import timm

from torch.utils.tensorboard import SummaryWriter

# databaseを作る
def make_dataframe(reference_path='train/',reference_meta_path='train_meta.json'):
    with open(reference_meta_path) as f:
        train_meta= json.load(f)
    df = pd.DataFrame()
    for dirname, _, filenames in os.walk(reference_path):
        df_temp = pd.DataFrame()
        if dirname != reference_path:
            print(dirname)
            label= os.path.basename(dirname)
            for key,value in  train_meta[label].items():
                df_temp[key]=[value]*len(filenames)

            df_temp["label"]=[label]*len(filenames)
            df_temp["img_name"]=filenames
            fun = lambda t:os.path.join(reference_path,label,t)
            df_temp["img_path"]=df_temp["img_name"].map(fun)
            df = pd.concat([df,df_temp])

    df = df.sort_values("img_path").reset_index(drop=True)
    return df


class TanachoSet(Dataset):
    def __init__(self, df, transform,class_num, target_class='label'):
        self.df = df
        self.transform = transform
        self.class_num  = class_num 
        self.target_class = target_class

    def __getitem__(self, index):
        img = self.__get_img(index)
        img = self.transform(img) #torch vision

        label = int(self.df[self.target_class].iloc[index])
        label_onehot = torch.eye(self.class_num)[label]

        return img, label_onehot

    def __len__(self):
        return self.df.shape[0]

    def __get_img(self,index):
        path = self.df['img_path'].iloc[index]
        img = Image.open(path) #torch vision
        #img = cv2.imread(path) #A
        return img


def set_dataloader(df_train,df_eval,transform,cfg):

    transform = transform
    target_class = cfg['target_class']
    class_num = cfg['class_num']
    # Dataset を作成する。
        
    train_dataset= TanachoSet(df_train, transform['train'], class_num , target_class)
    eval_dataset = TanachoSet(df_eval, transform['eval'], class_num , target_class)
      

    # DataLoader を作成する。
    train_batch_size = cfg['train_batch_size'] #128
    eval_batch_size = cfg['eval_batch_size'] #64
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, 
    num_workers=os.cpu_count(), pin_memory=True
    )
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size,shuffle=False, 
    num_workers=os.cpu_count(), pin_memory=True
    )

    data_loader = {'train':train_loader,
                'eval':eval_loader }

    return data_loader

def run(model,data_loader,device,optimizer,criterion,scaler,phase,epoch,log_embedding,metric_metric_learning):

    loss_value= 0
    score = 0
    num = 0
    mat_s = []
    meta_data_s = []

    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        model.eval()   # Set model to evaluate mode

    for inputs, labels in data_loader[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()    

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'): # これを書いておくとfalseの時にtensorの勾配の計算をしないで済む
            with torch.cuda.amp.autocast():
                # if metric_metric_learning:
                outputs = model(inputs, torch.argmax(labels, axis= 1))
                # else:
                #     outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            _, corrects = torch.max(labels, 1)

            #backward + optimize only if in training phase
            if phase == 'train':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif phase == 'eval':
                features = model.extract(inputs)
                mat_s.append(features)
                meta_data_s.append(corrects)

        loss_value += loss.item()
        num += len(labels)
        score += get_score(outputs.sigmoid(),labels)

        #score += torch.sum(preds == corrects).item()

    if (phase=='eval') and (log_embedding):
        mat = torch.cat(mat_s)
        meta_data = torch.cat(meta_data_s).to("cpu").detach().tolist().copy()
        tb_logger.add_embedding(
                                    mat=mat,
                                    metadata=meta_data,
                                    global_step=epoch
                                )
    if num:
        loss_value = loss_value/num
        score  = score/num

    return loss_value, score, preds, corrects, outputs

def train_model(model,data_loader,cfg,optimizer,criterion,scaler,scheduler,save_path):

    train_loss_s = []
    eval_loss_s = []
    eval_acc_s = []
    eval_loss = 0
    eval_acc = 0

    train_loss_min = 10e10
    eval_loss_min = 10e10
    eval_acc_max = -1
    num_epochs = cfg['epoch']
    fold  = cfg['fold']
    log_embedding = cfg['log_embedding']
    metric_metric_learning = cfg['metric_learning']

    if fold  != -1:
        phase_s = ['train','eval']
    else:
        phase_s = ['train']

    for epoch in range(num_epochs):
        print (f'Epoch [{(epoch+1)}/{num_epochs}]')
        
        
        for phase in phase_s:
            loss_value,score,preds,corrects,outputs= run(model,data_loader,cfg['device'],optimizer,criterion,scaler,phase,epoch,log_embedding,metric_metric_learning )
            if phase == 'train':
                train_loss = loss_value* cfg['train_batch_size']
            else:
                eval_loss = loss_value* cfg['eval_batch_size']
                eval_acc = score
        
        if cfg['scheduler'] != 'none':
            scheduler.step(epoch+1)

        if fold  != -1:
            #is_best_model = (eval_loss_min >eval_loss)
            is_best_model = (eval_acc_max<eval_acc)
        else:
            is_best_model = (train_loss_min >train_loss)

        if is_best_model:
            print('best model ever.')
            torch.save(model.state_dict(), save_path)

        train_loss_s.append(train_loss)
        eval_loss_s.append(eval_loss)
        eval_acc_s.append(eval_acc)    
        eval_acc_max = max(eval_acc_s)
        train_loss_min = min(train_loss_s)
        eval_loss_min = min(eval_loss_s)

        print (f'train_loss: {train_loss:.5f}, eval_loss: {eval_loss:.5f}, score: {eval_acc:.5f}')
        print(f'best score is map:{eval_acc_max:.5f}')
        #plot_loss(train_loss_s,eval_loss_s,eval_acc_s)

        tb_logger.add_scalar('Loss/train', train_loss, epoch)
        tb_logger.add_scalar('Loss/valid', eval_loss, epoch)
        tb_logger.add_scalar('score/valid', eval_acc, epoch)
        tb_logger.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)

        

    return train_loss_s, eval_loss_s, eval_acc_s
        

def set_seed(seed: int = 42):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        #torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False  # type: ignore
        #torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # type: ignore
        #torch.backends.cudnn.allow_tf32 = True

# t-sne を用いた可視化
# https://learnopencv.com/t-sne-for-feature-visualization/
def split_train_valid_df(df,cfg):

    # Split
    skf = StratifiedKFold(n_splits=cfg['fold_splits'], shuffle=True, random_state=cfg['seed'])
    for n, (train_index, val_index) in enumerate(skf.split(df, df[cfg['target_class']])):
        df.loc[val_index, "fold"] = int(n)
    df["fold"] = df["fold"].astype(int)

    fold = int(cfg['fold'])
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    return train_df, valid_df

def get_score(preds,labels):
    """AI is creating summary for get_score

    Args:
        preds ([type]): [[0.13, 0.01, 0.42, ..., 0.23, 0.98, 0.123],...,]] (n,122)の２次元配列
        labels ([type]): [[0.0, 0.0, 0.0, ..., 0.0, 1.0, 0.0],...,]] (n,122)の２次元配列

    Returns:
        float: map@10 × n
    """
    _, idx = torch.sort(preds, descending=True)
    sorted = torch.gather(labels, -1, idx)
    maxs = torch.argmax(sorted, dim=1) + 1
    scores = 1/maxs
    scores[scores < 0.1] = 0
    score = torch.sum(scores)

    return score.item()

if __name__ == '__main__':
    # nohup python train.py  -> ../log/log.txt &

    import datetime
    dt_now = datetime.datetime.now().strftime("%y%m%d_%H%M")    

    cfg = dict(
        # いろんな設定
        dt_now =  dt_now,
        class_num = 122,
        train_batch_size = 16,
        eval_batch_size = 64,
        epoch = 120,
        fold = -1,
        img_size = 224, #224
        target_class='label',
        fold_splits =5,
        seed =0,
        optimizer='adam', #'sgd', 'radam', 'adadelta', 'rmsprop'
        scheduler='cosine_warmup',#'cosine','step','none'
        lr = 1.2e-4,
        pre_train_model = 'swin_base_patch4_window7_224_in22k', #'tf_efficientnetv2_l_in21ft1k','swin_base_patch4_window7_224_in22k', 'resnetv2_101x1_bitm', 'tf_efficientnet_b7', 'convnext_large_in22ft1k'
        metric_learning = 'arcface', #'','adacos', 'arcface', 'sphereface', 'cosface'
        log_embedding = False,
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    para_s = [

        {
            'lr':1.4e-4,
            'train_batch_size': 16,
            'pre_train_model':'swin_base_patch4_window7_224_in22k'
        },
        {
            'lr':1.4e-4,
            'train_batch_size': 16,
            'pre_train_model':'tf_efficientnetv2_l_in21ft1k'
        },
        {
            'lr':1.4e-4,
            'train_batch_size': 16,
            'pre_train_model':'tf_efficientnet_b7'
        },
        # {
        #     'lr':0.7e-4,
        #     'train_batch_size': 8,
        #     'pre_train_model':'convnext_large_in22ft1k'
        # },
    ]

    for i, para in enumerate(para_s):
        
        cfg.update(para)
        exp_dir = os.path.join('../log',dt_now)
        cfg['log_path'] = os.path.join(exp_dir, str(i))

        set_seed(cfg['seed']) 
        transform  = get_transform(cfg['img_size'])
        with open('df.pkl', 'rb') as f:
            df = pickle.load(f)
        df_train, df_eval = split_train_valid_df(df, cfg)
        data_loader = set_dataloader(df_train, df_eval, transform, cfg)        
        model = get_model(cfg)

        metric_dict = {'Loss/train':0,
                        'Loss/valid':0,
                        'score/valid':0
                        }

        tb_logger = setup_logger(cfg,str(i),metric_dict)
        model.eval()
        test_image = torch.randn(1, 3, cfg['img_size'], cfg['img_size']).to(cfg['device'])
        test_image.to(cfg['device'])
        tb_logger.add_graph(model, test_image)
        save_config(cfg,cfg['log_path'])

        criterion = nn.CrossEntropyLoss()
        optimizer,model = get_optimizer(cfg, model)
        scheduler,optimizer = get_scheduler(cfg, optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        train_loss_s,eval_loss_s,eval_acc_s = train_model(model,
                                                        data_loader,
                                                        cfg,
                                                        optimizer,
                                                        criterion,
                                                        scaler,
                                                        scheduler,
                                                        os.path.join(cfg['log_path'],'best_model.pt')
                                                        )
