import os
import pickle
from skimage.transform import resize
import torch
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from torch.utils.data import Dataset,TensorDataset,DataLoader
from sklearn import preprocessing
from preprocess import trainprepro
from test_preprocess import test_prepro

class SENSOR_DATASET():
    def __init__(self, cfg):
        #super(SENSOR_DATASET, self).__init__()
        self.cfg = cfg
        self.sensor_length = cfg.sensor_length
        self.sensor_channel_size = cfg.sensor_channel_size
        self.image_channel_size = cfg.image_channel_size
        self.dataset_dir = cfg.dataset_dir
        self.num_instances = cfg.num_instances
        self.sensor_channel_size = cfg.sensor_channel_size
        self.testdataset_dir = cfg.testdataset_dir
        self.testnum_instances = cfg.testnum_instances
        #data_dir = 'G:\\毕业设计\\数据集\\速变数据\\traindata'

        self.image_channel_size = cfg.image_channel_size
        #train_x, test_x = trainprepro(d_path=self.dataset_dir, length=1000,number=2000, normal=True, rate=[0.9, 0.1], enc=True,enc_step=28)

        #train_x = np.expand_dims(train_x, axis=1)
        #test_x = np.expand_dims(test_x, axis=1)
        #np.savez(os.path.join(data_dir, 'shuju.npz'), train=train_x, valid=test_x)

        """

        data = np.load('trainnewnew.npz')
        test_X = data['valid']
        train_X = data['train']
        self.instance_idx_train = train_X.shape[0]
        self.instance_idx_test = test_X.shape[0]
        train_X = torch.from_numpy(train_X[:,:,:,:]).double()
        test_X = torch.from_numpy(test_X[:,:,:,:]).double()

        train_X = TensorDataset(train_X)
        test_X = TensorDataset(test_X)

        self.data_train = train_X
        self.data_test = test_X
        
        """






        #--test时候采用


        """


        data = np.load('trainnewnew.npz')

        test_X = data['valid']

        test_Y = []
        for i in range(test_X.shape[0]):
            test_Y.append(1)
        test_Y= np.asarray(test_Y)

        test_Y = torch.from_numpy(test_Y)




        self.instance_idx_Test = test_X.shape[0]
        test_X = torch.from_numpy(test_X).double()

        test_X = TensorDataset(test_X,test_Y)
        self.data_Test = test_X



        

        """

        test_X, test_Y = test_prepro(d_path=self.testdataset_dir, test_length=self.sensor_length, test_number=1000,
                                    normal=False)
        
        test_X = np.expand_dims(test_X, axis=1)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0.3,0.7),copy = False)
        for i in range(2000):
            test_X[i][0] = min_max_scaler.fit_transform(test_X[i][0])




        self.instance_idx_Test = test_X.shape[0]
        test_X = torch.from_numpy(test_X).double()

        test_X = TensorDataset(test_X,test_Y)
        self.data_Test = test_X

















































