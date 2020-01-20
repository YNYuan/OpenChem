import numpy as np
import pandas as pd 

from openchem.data.graph_data_layer import GraphDataset_Single, GraphDataset_Multi

class ACDataReader():
    def __init__(self, number_of_sample, train_ratio, random_seed=32):
        self.number_of_sample  = number_of_sample
        self.train_ratio = train_ratio
        self.random_seed = random_seed

    def random_index(self, length, train_size):
        indices = np.random.RandomState(seed=self.random_seed).permutation(length)
        cutNum = int(np.floor(length*train_size))
        train_idx, test_idx = indices[:cutNum], indices[cutNum:]
        return train_idx, test_idx

    def my_feature_split(self, matrix_path, train_idx, test_idx):
        matrix = np.load(matrix_path)
        train_matrix, test_matrix = matrix[train_idx,:,:], matrix[test_idx,:,:]
        return train_matrix, test_matrix

    def my_target_split(self, whole_path, train_idx, test_idx):
        name_file = pd.read_csv(whole_path)
        target = name_file['affinity'].values
        trainY = target[train_idx].reshape(-1,1)
        testY = target[test_idx].reshape(-1,1)
        return trainY, testY
    
    def get_dataset_single(self, xpath, zpath, nbrspath, nbrszpath, targetpath):
        train_idx, test_idx = self.random_index(self.number_of_sample, self.train_ratio)
        trainX, testX = self.my_feature_split(xpath, train_idx, test_idx)
        trainZ, testZ = self.my_feature_split(zpath, train_idx, test_idx)
        trainNbrs, testNbrs = self.my_feature_split(nbrspath, train_idx, test_idx)
        trainNbrs_Z, testNbrs_Z = self.my_feature_split(nbrszpath, train_idx, test_idx)
        trainY, testY = self.my_target_split(targetpath, train_idx, test_idx)
        train_dataset = GraphDataset_Single(trainX, trainZ, trainNbrs, trainNbrs_Z, trainY)
        test_dataset = GraphDataset_Single(testX, testZ, testNbrs, testNbrs_Z, testY)
        return train_dataset, test_dataset

    def get_dataset_multi(self, xpath_l, zpath_l, nbrspath_l, nbrszpath_l, xpath_r, zpath_r, 
    nbrspath_r, nbrszpath_r, xpath_c, zpath_c, nbrspath_c, nbrszpath_c, targetpath):
        train_idx, test_idx = self.random_index(self.number_of_sample, self.train_ratio)
        trainX_l, testX_l = self.my_feature_split(xpath_l, train_idx, test_idx)
        trainZ_l, testZ_l = self.my_feature_split(zpath_l, train_idx, test_idx)
        trainNbrs_l, testNbrs_l = self.my_feature_split(nbrspath_l, train_idx, test_idx)
        trainNbrs_Z_l, testNbrs_Z_l = self.my_feature_split(nbrszpath_l, train_idx, test_idx)
        trainX_r, testX_r = self.my_feature_split(xpath_r, train_idx, test_idx)
        trainZ_r, testZ_r = self.my_feature_split(zpath_r, train_idx, test_idx)
        trainNbrs_r, testNbrs_r = self.my_feature_split(nbrspath_r, train_idx, test_idx)
        trainNbrs_Z_r, testNbrs_Z_r = self.my_feature_split(nbrszpath_r, train_idx, test_idx)
        trainX_c, testX_c = self.my_feature_split(xpath_c, train_idx, test_idx)
        trainZ_c, testZ_c = self.my_feature_split(zpath_c, train_idx, test_idx)
        trainNbrs_c, testNbrs_c = self.my_feature_split(nbrspath_c, train_idx, test_idx)
        trainNbrs_Z_c, testNbrs_Z_c = self.my_feature_split(nbrszpath_c, train_idx, test_idx)
        trainY, testY = self.my_target_split(targetpath, train_idx, test_idx)
        train_dataset = GraphDataset_Multi(trainX_l, trainZ_l, trainNbrs_l, trainNbrs_Z_l, 
        trainX_r, trainZ_r, trainNbrs_r, trainNbrs_Z_r, trainX_c, trainZ_c, 
        trainNbrs_c, trainNbrs_Z_c, trainY)
        test_dataset = GraphDataset_Multi(testX_l, testZ_l, testNbrs_l, testNbrs_Z_l, 
        testX_r, testZ_r, testNbrs_r, testNbrs_Z_r, testX_c, testZ_c, testNbrs_c, 
        testNbrs_Z_c, testY)
        return train_dataset, test_dataset