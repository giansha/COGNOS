import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train", cache_dir=None): 
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()


        if cache_dir is None:
            self.cache_dir = os.path.join(root_path, 'psm_cache') 
        else:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        cache_file_name = f"psm_preprocessed_data_{os.path.basename(root_path)}.npz"
        self.cache_file_path = os.path.join(self.cache_dir, cache_file_name)

        if os.path.exists(self.cache_file_path):
            print(f"Loading data from cache: {self.cache_file_path}")
            cached_data = np.load(self.cache_file_path)
            self.train = cached_data['train']
            self.test = cached_data['test']
            self.val = cached_data['val']
            self.test_labels = cached_data['test_labels']
            print("Data loaded from cache successfully.")
        else:
            print(f"Cache not found ({self.cache_file_path}). Preprocessing data...")

            data = pd.read_csv(os.path.join(root_path, 'train.csv'))
            data = data.values[:, 1:]
            data = np.nan_to_num(data)
            self.scaler.fit(data)
            data = self.scaler.transform(data)

            test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
            test_data = test_data.values[:, 1:]
            test_data = np.nan_to_num(test_data)
            self.test = self.scaler.transform(test_data)

            self.train = data
            data_len = len(self.train)
            self.val = self.train[(int)(data_len * 0.8):]
            self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]

   
            np.savez_compressed(self.cache_file_path,
                                train=self.train,
                                test=self.test,
                                val=self.val,
                                test_labels=self.test_labels)
            print(f"Data preprocessed and saved to cache: {self.cache_file_path}")
       

        print("train shape:", self.train.shape)
        print("test shape:", self.test.shape)
        print("val shape:", self.val.shape)
        print("test_labels shape:", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train", cache_dir=None,discrete_channels=None): # 添加 cache_dir 参数
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        
        if cache_dir is None:
            self.cache_dir = os.path.join(root_path, 'msl_cache') 
        else:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        cache_file_name = f"msl_preprocessed_data_{os.path.basename(root_path)}.npz" 
        self.cache_file_path = os.path.join(self.cache_dir, cache_file_name)

        if os.path.exists(self.cache_file_path):
            print(f"Loading data from cache: {self.cache_file_path}")
            cached_data = np.load(self.cache_file_path)
            self.train = cached_data['train']
            self.test = cached_data['test']
            self.val = cached_data['val']
            self.test_labels = cached_data['test_labels']
            print("Data loaded from cache successfully.")
        else:
            print(f"Cache not found ({self.cache_file_path}). Preprocessing data...")
           
            data = np.load(os.path.join(root_path, "MSL_train.npy"))
            self.scaler.fit(data)
            data = self.scaler.transform(data)

            test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
            self.test = self.scaler.transform(test_data)

            self.train = data
            data_len = len(self.train)
            self.val = self.train[(int)(data_len * 0.8):]
            self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))

          
            np.savez_compressed(self.cache_file_path,
                                train=self.train,
                                test=self.test,
                                val=self.val,
                                test_labels=self.test_labels)
            print(f"Data preprocessed and saved to cache: {self.cache_file_path}")
      

        print("train shape:", self.train.shape)
        print("test shape:", self.test.shape)
        print("val shape:", self.val.shape)
        print("test_labels shape:", self.test_labels.shape)

        if discrete_channels is not None:
            self.train = np.delete(self.train, discrete_channels, axis=-1)
            self.test = np.delete(self.test, discrete_channels, axis=-1)
            self.val = np.delete(self.val, discrete_channels, axis=-1)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train", cache_dir=None, discrete_channels=None): # 添加 cache_dir 参数
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

     
        if cache_dir is None:
            self.cache_dir = os.path.join(root_path, 'smap_cache')
        else:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        cache_file_name = f"smap_preprocessed_data_{os.path.basename(root_path)}.npz" 
        self.cache_file_path = os.path.join(self.cache_dir, cache_file_name)

        if os.path.exists(self.cache_file_path):
            print(f"Loading data from cache: {self.cache_file_path}")
            cached_data = np.load(self.cache_file_path)
            self.train = cached_data['train']
            self.test = cached_data['test']
            self.val = cached_data['val']
            self.test_labels = cached_data['test_labels']
            print("Data loaded from cache successfully.")
        else:
            print(f"Cache not found ({self.cache_file_path}). Preprocessing data...")
           
            data = np.load(os.path.join(root_path, "SMAP_train.npy"))
            self.scaler.fit(data)
            data = self.scaler.transform(data)

            test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
            self.test = self.scaler.transform(test_data)

            self.train = data
            data_len = len(self.train)
            self.val = self.train[(int)(data_len * 0.8):]
            self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))

         
            np.savez_compressed(self.cache_file_path,
                                train=self.train,
                                test=self.test,
                                val=self.val,
                                test_labels=self.test_labels)
            print(f"Data preprocessed and saved to cache: {self.cache_file_path}")
       

        print("train shape:", self.train.shape)
        print("test shape:", self.test.shape)
        print("val shape:", self.val.shape)
        print("test_labels shape:", self.test_labels.shape)

        if discrete_channels is not None:
            self.train = np.delete(self.train, discrete_channels, axis=-1)
            self.test = np.delete(self.test, discrete_channels, axis=-1)
            self.val = np.delete(self.val, discrete_channels, axis=-1)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train", cache_dir=None,discrete_channels=None): # 添加 cache_dir 参数
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

      
        if cache_dir is None:
            self.cache_dir = os.path.join(root_path, 'smd_cache')
        else:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        cache_file_name = f"smd_preprocessed_data_{os.path.basename(root_path)}.npz" 
        self.cache_file_path = os.path.join(self.cache_dir, cache_file_name)

        if os.path.exists(self.cache_file_path):
            print(f"Loading data from cache: {self.cache_file_path}")
            cached_data = np.load(self.cache_file_path)
            self.train = cached_data['train']
            self.test = cached_data['test']
            self.val = cached_data['val']
            self.test_labels = cached_data['test_labels']
            print("Data loaded from cache successfully.")
        else:
            print(f"Cache not found ({self.cache_file_path}). Preprocessing data...")
            
            data = np.load(os.path.join(root_path, "SMD_train.npy"))
            self.scaler.fit(data)
            data = self.scaler.transform(data)

            test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
            self.test = self.scaler.transform(test_data)

            self.train = data
            data_len = len(self.train)
            self.val = self.train[(int)(data_len * 0.8):]
            self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

            
            np.savez_compressed(self.cache_file_path,
                                train=self.train,
                                test=self.test,
                                val=self.val,
                                test_labels=self.test_labels)
            print(f"Data preprocessed and saved to cache: {self.cache_file_path}")

        

            print("train shape:", self.train.shape)
            print("test shape:", self.test.shape)
            print("val shape:", self.val.shape)
            print("test_labels shape:", self.test_labels.shape)

        if discrete_channels is not None:
            self.train = np.delete(self.train, discrete_channels, axis=-1)
            self.test = np.delete(self.test, discrete_channels, axis=-1)
            self.val = np.delete(self.val, discrete_channels, axis=-1)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train", cache_dir=None):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

     
        if cache_dir is None:
            
            self.cache_dir = os.path.join(root_path, 'swat_cache')
        else:
            self.cache_dir = cache_dir

      
        os.makedirs(self.cache_dir, exist_ok=True)

        
        cache_file_name = f"swat_preprocessed_data_{os.path.basename(root_path)}.npz"
        self.cache_file_path = os.path.join(self.cache_dir, cache_file_name)

     
        if os.path.exists(self.cache_file_path):
            print(f"Loading data from cache: {self.cache_file_path}")
            
            cached_data = np.load(self.cache_file_path)
            self.train = cached_data['train']
            self.test = cached_data['test']
            self.val = cached_data['val']
            self.test_labels = cached_data['test_labels']
            print("Data loaded from cache successfully.")
        else:
            print(f"Cache not found ({self.cache_file_path}). Preprocessing data...")
         
            train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
            test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
            labels = test_data.values[:, -1:]
            train_data = train_data.values[:, :-1]
            test_data = test_data.values[:, :-1]

            self.scaler.fit(train_data)
            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)

            self.train = train_data
            self.test = test_data
            data_len = len(self.train)
            self.val = self.train[(int)(data_len * 0.8):]
            self.test_labels = labels
            
            np.savez_compressed(self.cache_file_path,
                                train=self.train,
                                test=self.test,
                                val=self.val,
                                test_labels=self.test_labels)
            print(f"Data preprocessed and saved to cache: {self.cache_file_path}")
       

        print("train shape:", self.train.shape)
        print("test shape:", self.test.shape)
        print("val shape:", self.val.shape)
        print("test_labels shape:", self.test_labels.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])






class UCRAnomalyloader(Dataset):
    def __init__(self,args,root_path,win_size, data_path, train_length, border_1, border_2, step=1, flag="train", percentage=1):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        # 1.read data
        data_path = os.path.join(root_path, data_path)
        data = self.read_data(data_path)
        label = np.zeros(data.shape[0])
        label[border_1:border_2] = 1
        # 2.train
        train_data = data[:train_length]
        train_label = label[:train_length]
        # 3.test
        test_data = data[train_length:]
        test_label = label[train_length:]
        # 4.process
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if flag == "init":
            self.init = train_data
            self.init_label = train_label
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end * (1 - percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label

    @staticmethod
    def read_data(dataset_file_path):
        data_list = []
        try:
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    data_list.append(data_line)
            data = np.stack(data_list, 0)
        except ValueError:
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line[0].split()])
            data = data_line
            data = np.expand_dims(data, axis=1)
        return data

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "init":
            return (self.init.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index: index + self.win_size]), np.float32(
                self.train_label[index: index + self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index: index + self.win_size]), np.float32(
                self.val_label[index: index + self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        elif self.flag == "init":
            return np.float32(self.init[index: index + self.win_size]), np.float32(
                self.init_label[index: index + self.win_size])
        else:
            return np.float32(self.test[
                                  index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])


class TrainSegLoader(Dataset):
    def __init__(self, data_path, train_length, win_size, step=1, flag="train", percentage=1, discrete_channels=None):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        # 1.read data
        data = self.read_data(data_path)
        # 2.train
        train_data = data.iloc[:train_length, :]
        train_data, train_label = (
            train_data.loc[:, train_data.columns != "label"].to_numpy(),
            train_data.loc[:, ["label"]].to_numpy(),
        )
        # 3.test
        test_data = data.iloc[train_length:, :]
        test_data, test_label = (
            test_data.loc[:, test_data.columns != "label"].to_numpy(),
            test_data.loc[:, ["label"]].to_numpy(),
        )
        # 4.process
        if discrete_channels is not None:
            train_data = np.delete(train_data, discrete_channels, axis=-1)
            test_data = np.delete(test_data, discrete_channels, axis=-1)

        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if flag == "init":
            self.init = train_data
            self.init_label = train_label
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end * (1 - percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label

    @staticmethod
    def read_data(path: str, nrows=None) -> pd.DataFrame:
        data = pd.read_csv(path)
        label_exists = "label" in data["cols"].values
        all_points = data.shape[0]
        columns = data.columns
        if columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()
        is_univariate = n_points == all_points
        n_cols = all_points // n_points
        df = pd.DataFrame()
        cols_name = data["cols"].unique()
        if columns[0] == "date" and not is_univariate:
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif columns[0] != "date" and not is_univariate:
            col_data = {
                cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        elif columns[0] == "date" and is_univariate:
            df["date"] = data.iloc[:, 0]
            df[cols_name[0]] = data.iloc[:, 1]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        else:
            df[cols_name[0]] = data.iloc[:, 0]
        if label_exists:
            last_col_name = df.columns[-1]
            df.rename(columns={last_col_name: "label"}, inplace=True)
        if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
            df = df.iloc[:nrows, :]
        return df

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "init":
            return (self.init.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index, eps=1):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index: index + self.win_size]), np.float32(
                self.train_label[index: index + self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index: index + self.win_size]), np.float32(
                self.val_label[index: index + self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        elif self.flag == "init":
            return np.float32(self.init[index: index + self.win_size]), np.float32(
                self.init_label[index: index + self.win_size])
        else:
            return np.float32(self.test[
                                  index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])
