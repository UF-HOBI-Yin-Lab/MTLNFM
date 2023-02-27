import numpy as np
import torch.utils.data

class IPD_MA_MTL(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        ## 0-52, feature fields; 52-55, targets; 55-78 numerical_features scalar
        data = np.load(dataset_path,allow_pickle=True)
        self.features = data[:,0:52].astype(int)
        self.targets = data[:,52:55].astype(int)
        self.mask_value = data[:,55:].astype(float)
        self.mask = np.array([1,17,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,49])
            
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.targets[index],self.mask,self.mask_value[index]
    
class IPD_MA_STL(torch.utils.data.Dataset):
    def __init__(self, dataset_path,task_number):
        ## 0-52, feature fields; 52-55, targets; 55-78 numerical_features scalar
        data = np.load(dataset_path,allow_pickle=True)
        self.features = data[:,0:52].astype(int)
        self.targets = data[:,52:55].astype(int)
        self.mask_value = data[:,55:].astype(float)
        self.targets = self.targets[:,task_number]
        self.mask = np.array([1,17,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,49])
            
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.targets[index],self.mask,self.mask_value[index]
    
class IPD_MA_MTL_for2(torch.utils.data.Dataset):
    def __init__(self, dataset_path,task_number):
        ## 0-52, feature fields; 52-55, targets; 55-78 numerical_features scalar
        data = np.load(dataset_path,allow_pickle=True)
        self.features = data[:,0:52].astype(int)
        self.targets = data[:,52:55].astype(int)
        self.mask_value = data[:,55:].astype(float)
        self.targets = np.concatenate((self.targets[:,task_number[0]].reshape(self.targets.shape[0],-1)
                                       ,self.targets[:,task_number[1]].reshape(self.targets.shape[0],-1)),axis=1)
        self.mask = np.array([1,17,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,49])
            
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.targets[index],self.mask,self.mask_value[index]
    
class Load_saved_data_STL(torch.utils.data.Dataset):
    def __init__(self, dataset_path,task_number):
        ## 0-52, feature fields; 52-55, targets; 55-78 numerical_features scalar
        data = np.load(dataset_path,allow_pickle=True)
        self.features = []
        self.targets = []
        self.mask_value = []
        self.mask = np.array([1,17,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,49])

        for i in data:
            self.features.append(i[0])
            self.targets.append(i[1])
            self.mask_value.append(i[3])

        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        self.mask_value = np.array(self.mask_value)
        self.targets = self.targets[:,task_number]

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.targets[index],self.mask,self.mask_value[index]
    
class Load_saved_data_MTLfor2(torch.utils.data.Dataset):
    def __init__(self, dataset_path,task_number):
        ## 0-52, feature fields; 52-55, targets; 55-78 numerical_features scalar
        data = np.load(dataset_path,allow_pickle=True)
        self.features = []
        self.targets = []
        self.mask_value = []
        self.mask = np.array([1,17,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,49])

        for i in data:
            self.features.append(i[0])
            self.targets.append(i[1])
            self.mask_value.append(i[3])

        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        self.mask_value = np.array(self.mask_value)
        
        self.targets = np.concatenate((self.targets[:,task_number[0]].reshape(self.targets.shape[0],-1)
                                       ,self.targets[:,task_number[1]].reshape(self.targets.shape[0],-1)),axis=1)
            
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.targets[index],self.mask,self.mask_value[index]

class Load_saved_data_MTL(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        ## 0-52, feature fields; 52-55, targets; 55-78 numerical_features scalar
        data = np.load(dataset_path,allow_pickle=True)
        self.features = []
        self.targets = []
        self.mask_value = []
        self.mask = np.array([1,17,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,49])

        for i in data:
            self.features.append(i[0])
            self.targets.append(i[1])
            self.mask_value.append(i[3])

        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        self.mask_value = np.array(self.mask_value)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.targets[index],self.mask,self.mask_value[index]