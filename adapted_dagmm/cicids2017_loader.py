import os
import numpy as np
import pandas as pd

def get_train_ds(data_folder):
    url_data = os.path.join(data_folder, 'flow_based/Monday-WH-generate-labeled.csv')
    df = pd.read_csv(url_data)
    feats = df.iloc[:,8:]
    ds_port = df.iloc[:,5]
    df = pd.concat([ds_port,feats],axis=1)
    # print(df.columns.values)
    all_feats = df.iloc[:,:-1].astype(np.float32).values
    known_data_IDs =(np.any(np.isinf(all_feats),axis=1) + np.any(np.isnan(all_feats),axis=1))==False
    x_train = all_feats[known_data_IDs]
    
    y_train = df.iloc[:,-1].values
    y_train[y_train=='BENIGN']=0.
    y_train = y_train.astype(np.float32)
    y_train = y_train[known_data_IDs]
    
    # print(x_train.shape,y_train.shape)
    
    train_min = np.min(x_train,axis=0)
    train_max = np.max(x_train,axis=0)
    
    x_train  = (x_train - train_min)/(train_max - train_min + 1e-6)
    
    return x_train





def get_test_set(data_folder):
    test_files = [data_folder+'/flow_based/Tuesday-WH-generate-labeled.csv',
                data_folder+'/flow_based/Wednesday-WH-generate-labeled.csv',
                data_folder+'/flow_based/Thursday-WH-generate-labeled.csv',
                data_folder+'/flow_based/Friday-WH-generate-labeled.csv']

    # test_files = [data_folder+'/flow_based/Thursday-WH-generate-labeled.csv',
    #               data_folder+'/flow_based/Tuesday-WH-generate-labeled.csv']
    
    train_min = np.load(data_folder+'/flow_based/x_train_meta/train_min.npy')
    train_max = np.load(data_folder+'/flow_based/x_train_meta/train_max.npy')
    
    x_test_all = []
    y_test_all = []
    all_label_set = []
    for i in range(len(test_files)):
        # print (i,test_files[i])
        url_data = test_files[i]
        df = pd.read_csv(url_data)

        feats = df.iloc[:,8:]
        ds_port = df.iloc[:,5]
        df = pd.concat([ds_port,feats],axis=1)

        labels = df.iloc[:,-1].values
        label_set = set(labels)
        all_label_set.append(label_set)

        all_feats = df.iloc[:,:-1].astype(np.float64).values
        known_data_IDs =(np.any(np.isinf(all_feats),axis=1) + np.any(np.isnan(all_feats),axis=1))==False
        x_test = all_feats[known_data_IDs]
        y_test = df.iloc[:,-1].values
        y_test = y_test[known_data_IDs]
        x_test = (x_test - train_min)/(train_max - train_min+1e-6)
        x_test_all.append(x_test)
        y_test_all.append(y_test)
    x_test = np.concatenate(x_test_all,axis=0).astype(np.float32)
    y_test = np.concatenate(y_test_all,axis=0)
    
    return x_test, y_test
    
    


class CICIDS2017Loader(object):
    def __init__(self, data_path, mode="train"):
        x_train = get_train_ds(data_path)
        y_train = np.zeros(len(x_train))

        if mode == "train_only":
            self.mode="train"
            x_test = None
            y_test = None
        else:
            self.mode=mode
            x_test, y_test = get_test_set(data_path)
            self.real_test_labels = y_test

            y_test = (y_test != 'BENIGN').astype(np.float32)

        self.train = x_train
        self.train_labels = y_train
        self.test = x_test
        self.test_labels = y_test
    
    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])