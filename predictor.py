from lib import get_transform, save_config, load_config,make_dataframe , split_train_valid_df
import os
import glob
import torch
from models import get_model
from PIL import Image
import numpy as np
import pandas as pd
import pickle
from scipy import spatial
from sklearn.preprocessing import normalize


class ScoringService(object):
    @classmethod
    def get_model(
            cls, 
            model_path='../model/', 
            reference_path=None, 
            reference_meta_path=None, 
            is_make_train_embeddings=False
        ):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            reference_path (str): Path to the reference data.
            reference_meta_path (str): Path to the meta data.

        Returns:
            bool: The return value. True for success, False otherwise.
        """

        try:
            cls.model_s, cls.cfg_s  = load_model(model_path)

            if is_make_train_embeddings:
                train_embedding_s, train_label_s = [],[]
            else:
                train_embedding_s, train_label_s = cls.load_train_embeddings()

            if reference_path:
                ref_embedding_s, ref_label_s = cls.get_ref_embeddings(reference_path, reference_meta_path)
            else:
                ref_embedding_s, ref_label_s = [],[]
            #ref_embedding_s, ref_label_s = [],[]
            
            label_s  = train_label_s + ref_label_s
            embedding_s = train_embedding_s + ref_embedding_s
            
            embedding_s = normalize(np.array(embedding_s), axis=1)
            cls.embedding_s = embedding_s 
            cls.label_s = label_s

            return True
        except:
            return False

    @classmethod
    def get_el(cls):
        return cls.embedding_s, cls.label_s

    @classmethod
    def load_train_embeddings(cls):
        with open('embedding_s.pkl', 'rb') as f:
            embedding_s= pickle.load(f)
        with open('label_s.pkl', 'rb') as f:
            label_s= pickle.load(f)

        return list(embedding_s), label_s

    @classmethod
    def servey(cls):
        
        df = make_dataframe(reference_path='../train/',reference_meta_path='../train_meta.json')
        train_df, valid_df= split_train_valid_df(df,cls.cfg_s[0])        

    @classmethod
    def get_ref_embeddings(cls, reference_path, reference_meta_path):

        df = make_dataframe(reference_path, reference_meta_path)
        #train_df, valid_df = split_train_valid_df(df,cfg)
        embedding_s = []
        label_s = []
        for index,row in df.iterrows():
            img_path = row['img_path']
            embedding= cls.get_embeddings(img_path)
            label = row['label']
            embedding_s.append(embedding)
            label_s.append(label)
        
        return embedding_s, label_s

    @classmethod
    def get_embeddings(cls, img_path):

        img_base = Image.open(img_path)

        embeddings = []
        with torch.no_grad():
            for i, model in enumerate(cls.model_s):
                transform = get_transform(cls.cfg_s[i]['img_size'])
                img = transform['eval'](img_base)
                img = img.unsqueeze(0).to('cpu')
                model.eval()
                embeddings.append(model.extract(img).numpy()[0])

        embedding = np.mean(embeddings, axis=0)
        return embedding

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
                transform = get_transform(cls.cfg_s[i]['img_size'])
                img = transform['eval'](img_base)
                img= img.unsqueeze(0).to('cpu')
                model.eval()
                #preds.append(model(img).sigmoid().numpy()[0])
                preds.append(model(img).numpy()[0])


        pred = np.mean(preds, axis=0)
        df = pd.DataFrame()
        df['output']=pred
        df['label']=['{:03}'.format(i) for i in range(len(pred))]
        df = df.sort_values('output', ascending=False)

        # make prediction
        label_s = df['label'].to_list()[:10]
        pred_s = df['output'].to_list()[:10]

        print(pred_s[0])
        if pred_s[0]<3.5:
            output = cls.predict_post( input)
        else:
            output = {sample_name: label_s}
        return output 

    @classmethod
    def predict_post(cls, input):
        """Predict method

        Args:
            input (str): path to the image you want to make inference from

        Returns:
            dict: Inference for the given input.
        """
        top = False
        # load an image and get the file name
        sample_name = os.path.basename(input).split('.')[0]

        embedding = cls.get_embeddings(input)
        embedding = embedding[np.newaxis]
        embedding = normalize(embedding, axis=1)
        distances = spatial.distance.cdist(
                                            embedding, 
                                            cls.embedding_s, 
                                            'cosine'
                                            ) # 0に近いほど似てる

        df = pd.DataFrame()
        df['output']=distances[0]
        df['label']=cls.label_s
        df = df.sort_values('output')

        distance_s = []
        for i in set(df['label']):
            query = df['label'] == i
            distance = np.mean(df['output'][query])
            distance_s.append(distance)
        
        if top:
            pred_s = []
            # make prediction
            for index,row in df.iterrows():
                if not(row['label'] in pred_s ):
                    pred_s.append(row['label'])
            pred_s = pred_s[:10]

        else:
            df_new = pd.DataFrame()
            df_new["label"] = list(set(df["label"]))
            df_new['distance'] = distance_s
            df_new = df_new.sort_values('distance')
            pred_s = list(df_new["label"][0:10])

        # pred_s = []
        # # make prediction
        # for index,row in df.iterrows():
        #     if not(row['label'] in pred_s ):
        #         pred_s.append(row['label'])
        # pred_s = pred_s[:10]
        

        # make output
        output = {sample_name: pred_s}
        return output


def load_model(model_path):
    """
    load some model(s)
    """

    fdir = os.path.join(model_path,"*/*")
    model_s = []
    cfg_s = []
    fdir_s = glob.glob(fdir)
    for fdir in fdir_s:
        fpath = os.path.join(fdir,'cfg.json')
        cfg = load_config(fpath)
        cfg['device'] = 'cpu'
        model = get_model(cfg, pre_train=False)
        fpath = os.path.join(fdir,'best_model.pt')
        model.load_state_dict(torch.load(fpath, map_location=torch.device(cfg['device'])))
        
        model_s.append(model)
        cfg_s.append(cfg)

    return model_s, cfg_s

def make_train_pkl():
    ScoringService.get_model(
        model_path='../model/', 
        reference_path='../train', 
        reference_meta_path='../train_meta.json', 
        is_make_train_embeddings=True
    ) 

    embedding_s, label_s = ScoringService.get_el()

    with open('embedding_s.pkl', 'wb') as f:
        pickle.dump(embedding_s , f)
    with open('label_s.pkl', 'wb') as f:
        pickle.dump(label_s , f)

    print(ScoringService.predict('./0.jpg'))

if __name__=='__main__':
    make_train_pkl()
    ScoringService.get_model(
        model_path='../model/', 
        reference_path='../dog', 
        reference_meta_path='../dog/ref_meta.json', 
        is_make_train_embeddings=False
    ) 

    print(ScoringService.predict('./0.jpg'))
    print(ScoringService.predict('./123.jpg'))


