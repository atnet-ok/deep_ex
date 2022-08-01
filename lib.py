from torchvision import transforms
import json
import os
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from glob import glob
from pathlib import Path

def save_config(cfg,path):
    with open(os.path.join(path,'cfg.json'), mode="w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def load_config(path):
    with open(path, mode="r") as f:
        cfg = json.load(f)
    return cfg

def setup_logger(cfg,cond,metric_dict):
    from torch.utils.tensorboard import SummaryWriter

    exp_dir = str(Path(cfg['log_path']).parent)
    tb_logger = SummaryWriter(log_dir=exp_dir)
    tb_logger.add_hparams(hparam_dict=cfg, metric_dict=metric_dict , run_name=cond)
    tb_logger.close()
    fpath_s = glob(exp_dir +"/events.out.*")
    for fpath in fpath_s:
        os.remove(fpath )
    tb_logger = SummaryWriter(log_dir=cfg['log_path'])
    return tb_logger



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

def get_transform(img_size):

    transform  = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(
                degrees=[-15, 15], translate=(0.1, 0.1), scale=(0.5, 1.5)
            ),
            transforms.RandomAutocontrast(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2,p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=0,p=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize(round(img_size*256/224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return  transform 

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

