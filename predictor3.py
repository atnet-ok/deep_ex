from lib import get_transform, save_config, load_config, split_train_valid_df,make_dataframe
import os
import glob
import torch
from models import get_model
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn as nn


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path='../model/', reference_path=None, reference_meta_path=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            reference_path (str): Path to the reference data.
            reference_meta_path (str): Path to the meta data.

        Returns:
            bool: The return value. True for success, False otherwise.
        """

        try:
            cls.model_s, cls.transform_s  = load_model()
            return True
        except:
            return False


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input (str): path to the image you want to make inference from

        Returns:
            dict: Inference for the given input.
        """

        # load an image and get the file name
        sample_name = os.path.basename(input).split('.')[0]

        img_path = input
        img_base = Image.open(img_path)

        preds = []
        with torch.no_grad():
            for i, model in enumerate(cls.model_s):

                img = cls.transform_s[i]['eval'](img_base)
                img= img.unsqueeze(0).to('cpu')
                model.eval()
                preds.append(model(img).sigmoid().numpy()[0])
                #preds.append(model(img).numpy()[0])


        pred = np.mean(preds, axis=0)
        df = pd.DataFrame()
        df['output']=pred
        df['label']=['{:03}'.format(i) for i in range(len(pred))]
        df = df.sort_values('output', ascending=False)

        # make prediction
        label_s = df['label'].to_list()[:10]
        pred_s = df['output'].to_list()[:10]

        # make output
        output = {sample_name: label_s}
        return output #,pred_s

    @classmethod
    def servey(cls):

        df = make_dataframe(reference_path='../train/',reference_meta_path='../train_meta.json')
        train_df, valid_df = split_train_valid_df(df,cfg)        

def load_model():
    """
    load some model(s)
    """

    fdir = "../model/*/*"
    model_s = []
    transform_s = []
    fdir_s = glob.glob(fdir)
    for fdir in fdir_s:
        fpath = os.path.join(fdir,'cfg.json')
        cfg = load_config(fpath)
        cfg['device'] = 'cpu'
        model = get_model(cfg, pre_train=False)
        fpath = os.path.join(fdir,'best_model.pt')
        model.load_state_dict(torch.load(fpath, map_location=torch.device('cpu')))
        model_s.append(model)
        transform = get_transform(cfg['img_size'])
        transform_s.append(transform)
    return model_s, transform_s

df = make_dataframe(reference_path='../train/',reference_meta_path='../train_meta.json')
split_train_valid_df(df,cfg)



