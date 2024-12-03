#coding:utf8
import os
from torch.utils import data
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
import numpy as np
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")

class Task1Data(data.Dataset):

    def __init__(self, is_train=True, is_test = False,shuffle_seed=42):
        self.root = '/home/yyang/yang/fmri_tokenizer/data'
        df = pd.read_csv(f"{self.root}/use_adhd.csv")
        self.aim_len = 64
        if is_train is True:
            use_index = df[df['is_train']==1].index
            self.names = np.array(list(df.iloc[use_index]['new_name']))
            self.dx = np.array(list(df.iloc[use_index]['new_dx'])).astype(int)
            print(f"Finding Training files: {len(self.names)}")
        else:
            use_index = df[df['is_train']==0].index

            site_data = np.array(list(df.iloc[use_index]['site']))
            names = np.array(list(df.iloc[use_index]['new_name']))
            dx = np.array(list(df.iloc[use_index]['new_dx'])).astype(int)

            test_length = int(0.15 * len(df))
            split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)

            for valid_index, test_index in split2.split(names, site_data):
                test_names, test_dx = names[test_index], dx[test_index]
                val_names, val_dx = names[valid_index], dx[valid_index]
            if is_test is True:
                self.names = test_names
                self.dx = test_dx
                print(f"Finding Testing files: {len(self.names)}")
                print(f"0/1: {len(self.names[self.dx==0])}/{len(self.names[self.dx==1])}")
            else:
                self.names = val_names
                self.dx = val_dx
                print(f"Finding Validation files: {len(self.names)}")
                print(f"0/1: {len(self.names[self.dx==0])}/{len(self.names[self.dx==1])}")
            # TODO: is_test

    def pad_or_cut(self,dat):
        roi_num, time_len = dat.shape
        if time_len < self.aim_len:
            ret = np.pad(dat,((0,0),(0,self.aim_len - time_len)))
        else:
            rand = np.random.randint(time_len - self.aim_len) // 4 * 4
            ret = dat[:, rand: rand + self.aim_len]
        return ret.T
    def __getitem__(self,index):
        file_name = f"{self.root}/run4_adhd/sch_{self.names[index]}.npy"
        dat = np.array(np.load(file_name))

        use_dat = self.pad_or_cut(dat)
        dx = self.dx[index]
        return use_dat, dx # (T, 100)

    def __len__(self):
        return len(self.names)

if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
        
    train_dataset = Task1Data()
    train_loader = DataLoader(train_dataset, 32,num_workers=0)
