# -*- coding:utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torchvision import transforms
from PIL import Image
from sklearn import preprocessing
import random

def KAT_Label(mode_path):
    filelist = os.listdir(mode_path)
    classlist = []
    for file in filelist:
        classlist.append(file.split('----')[0])
    classlist = np.unique(classlist).tolist()
    idx_dict = {}
    for i, name in enumerate(classlist):
        idx_dict[name] = i
    return idx_dict

def TrainLoadTensor(mode_path, idx_dict):
    Filename_list = os.listdir(mode_path)
    Filename_list.sort(key=lambda x : (x.split('----')[0], x.split('----')[-1].split('.')[0]))
    idxlist = []
    for file in Filename_list:
        idxlist.append(file.split('.')[0])
    idxlist = np.unique(idxlist).tolist()
    df_dict = {}
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    for file in idxlist:
        matname = file+'.mat'
        matpath = os.path.join(mode_path, matname)
        jpgname = file+'.jpg'
        jpgpath = os.path.join(mode_path, jpgname)
        df_dict[file] = loadmat(matpath)
        img = Image.open(jpgpath)
        df_dict[file]['Time_Frequency'] = img_transform(img)
    for _, k in df_dict.items():
        del k['__header__']
        del k['__version__']
        del k['__globals__']
    df = pd.DataFrame.from_dict(df_dict).T
    df = df.reset_index().rename({'index':'Index'}, axis=1)
    prearray = np.hstack([np.hstack(df['DE_time'].values).T, np.hstack(df['FFT_data'].values).T])
    scaler = preprocessing.StandardScaler()
    scaler.fit(prearray)
    prearray = scaler.transform(prearray)
    DElist, FFTlist = [], []
    DEarray, FFTarray = prearray[:, :512], prearray[:, 512:]
    for i in range(DEarray.shape[0]):
        DElist.append(DEarray[i])
    for j in range(FFTarray.shape[0]):
        FFTlist.append(FFTarray[j])
    predf = pd.concat([pd.Series(DElist), pd.Series(FFTlist)], axis=1, join='outer')
    predf = predf.rename({0:'DE_time', 1:'FFT_data'}, axis=1)
    df_fusion = pd.concat([df.drop(['DE_time', 'FFT_data'], axis=1), predf], axis=1, join='outer')
    df_fusion = df_fusion.assign(label=df_fusion['Index'].apply(lambda x: idx_dict[x.split('----')[0]]))
    return df_fusion, scaler

def TestLoadTensor(mode_path, idx_dict, scaler):
    Filename_list = os.listdir(mode_path)
    Filename_list.sort(key=lambda x : (x.split('----')[0], x.split('----')[-1].split('.')[0]))

    idxlist = []
    for f in Filename_list:
        # 可选：只看 .mat 或 .jpg，避免 Thumbs.db 之类
        if not (f.endswith(".mat") or f.endswith(".jpg")):
            continue
        idxlist.append(f.split('.')[0])
    idxlist = np.unique(idxlist).tolist()

    df_dict = {}
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    for file in idxlist:
        matname = file+'.mat'
        matpath = os.path.join(mode_path, matname)
        jpgname = file+'.jpg'
        jpgpath = os.path.join(mode_path, jpgname)
        df_dict[file] = loadmat(matpath)
        img = Image.open(jpgpath)
        df_dict[file]['Time_Frequency'] = img_transform(img)
    for _, k in df_dict.items():
        del k['__header__']
        del k['__version__']
        del k['__globals__']
    df = pd.DataFrame.from_dict(df_dict).T
    df = df.reset_index().rename({'index':'Index'}, axis=1)
    prearray = np.hstack([np.hstack(df['DE_time'].values).T, np.hstack(df['FFT_data'].values).T])
    prearray = scaler.transform(prearray)
    DElist, FFTlist = [], []
    DEarray, FFTarray = prearray[:, :512], prearray[:, 512:]
    for i in range(DEarray.shape[0]):
        DElist.append(DEarray[i])
    for j in range(FFTarray.shape[0]):
        FFTlist.append(FFTarray[j])
    predf = pd.concat([pd.Series(DElist), pd.Series(FFTlist)], axis=1, join='outer')
    predf = predf.rename({0:'DE_time', 1:'FFT_data'}, axis=1)
    df_fusion = pd.concat([df.drop(['DE_time', 'FFT_data'], axis=1), predf], axis=1, join='outer')
    df_fusion = df_fusion.assign(label=df_fusion['Index'].apply(lambda x: idx_dict[x.split('----')[0]]))

    return df_fusion

def Randomselct(df, k, n, q):
    clist = df['label'].unique().tolist()
    TF_support, DE_support, FFT_support = [], [], []
    TF_query, DE_query, FFT_query = [], [], []
    random.shuffle(clist)
    select_k = np.random.choice(clist, size=k, replace=False)
    for it in select_k:
        df1 = df[df['label'].isin([it])]
        df_sample = df1.sample(n=n+q, replace=False)
        df_support, df_query = df_sample[:n], df_sample[n:]
        TF_support.append(torch.stack(df_support['Time_Frequency'].values.tolist()))
        DE_support.append(torch.from_numpy(np.stack(df_support['DE_time'].values.tolist())).unsqueeze(dim=1))
        FFT_support.append(torch.from_numpy(np.stack(df_support['FFT_data'].values.tolist())).unsqueeze(dim=1))

        TF_query.append(torch.stack(df_query['Time_Frequency'].values.tolist()))
        DE_query.append(torch.from_numpy(np.stack(df_query['DE_time'].values.tolist())).unsqueeze(dim=1))
        FFT_query.append(torch.from_numpy(np.stack(df_query['FFT_data'].values.tolist())).unsqueeze(dim=1))
    TFs = torch.stack(TF_support).to(torch.float32)  # TENSOR(k ,n, 3, 224, 224)
    DEs = torch.stack(DE_support).to(torch.float32)  # TENSOR(k, n, 1, 512)
    FFTs = torch.stack(FFT_support).to(torch.float32)  # TENSOR(k, n, 1, 256)
    TFq = torch.stack(TF_query).to(torch.float32)  # TENSOR(k, q, 3, 224, 224)
    DEq = torch.stack(DE_query).to(torch.float32)  # TENSOR(k, q, 1, 512)
    FFTq = torch.stack(FFT_query).to(torch.float32)  # TENSOR(K, q, 1, 256)

    return TFs, DEs, FFTs, TFq, DEq, FFTq

class TestSampler:
    def __init__(self, opt, scaler):
        super(TestSampler, self).__init__()
        self.data_dir = opt.data_dir
        self.test_domain = opt.test_domain
        self.k = opt.k_val
        self.n = opt.n_val
        self.q = opt.q_val
        mode_path = os.path.normpath(os.path.join(self.data_dir, self.test_domain))
        idx_dict = KAT_Label(mode_path=mode_path)
        self.df_fusion = TestLoadTensor(mode_path=mode_path, idx_dict=idx_dict, scaler=scaler)

    def __iter__(self):
        return self

    def __next__(self):
        TFs, DEs, FFTs, TFq, DEq, FFTq = Randomselct(self.df_fusion, k=self.k, n=self.n, q=self.q)
        TFs, DEs, FFTs = TFs.cuda(), DEs.cuda(), FFTs.cuda()
        TFq, DEq, FFTq = TFq.cuda(), DEq.cuda(), FFTq.cuda()
        return TFs, DEs, FFTs, TFq, DEq, FFTq

class TrainSampler:
    def __init__(self, opt):
        super(TrainSampler, self).__init__()
        self.data_dir = opt.data_dir
        self.train_domain = opt.train_domain
        self.k = opt.k_train
        self.n = opt.n_train
        self.q = opt.q_train
        mode_path = os.path.normpath(os.path.join(self.data_dir, self.train_domain))
        idx_dict = KAT_Label(mode_path=mode_path)
        self.df_fusion, self.scaler = TrainLoadTensor(mode_path=mode_path, idx_dict=idx_dict)

    def __iter__(self):
        return self

    def __next__(self):
        TFs, DEs, FFTs, TFq, DEq, FFTq = Randomselct(df=self.df_fusion, k=2*self.k, n=self.n, q=self.q)
        ps_TFs, ps_DEs, ps_FFTs, ps_TFq, ps_DEq, ps_FFTq = TFs[:self.k], DEs[:self.k], FFTs[:self.k], TFq[:self.k], DEq[:self.k], FFTq[:self.k]
        ps_TFs, ps_DEs, ps_FFTs, ps_TFq, ps_DEq, ps_FFTq = ps_TFs.cuda(), ps_DEs.cuda(), ps_FFTs.cuda(), ps_TFq.cuda(), ps_DEq.cuda(), ps_FFTq.cuda()

        pu_TFs, pu_DEs, pu_FFTs, pu_TFq, pu_DEq, pu_FFTq = TFs[self.k:], DEs[self.k:], FFTs[self.k:], TFq[self.k:], DEq[self.k:], FFTq[self.k:]
        pu_TFs, pu_DEs, pu_FFTs, pu_TFq, pu_DEq, pu_FFTq = pu_TFs.cuda(), pu_DEs.cuda(), pu_FFTs.cuda(), pu_TFq.cuda(), pu_DEq.cuda(), pu_FFTq.cuda()
        return ps_TFs, ps_DEs, ps_FFTs, ps_TFq, ps_DEq, ps_FFTq, pu_TFs, pu_DEs, pu_FFTs, pu_TFq, pu_DEq, pu_FFTq

