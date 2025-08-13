# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from einops import rearrange


class DataScalerStander:
    def __init__(self, y, config):
        self.config = config
        y = self.__check_input__(y)
        self.original_shape = None
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            self.original_shape = y.shape
            y = y.reshape(-1, y.shape[-1])
        train_data = y[:int(len(y) * self.config.density)].astype(np.float32)
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

    def transform(self, y):
        y = self.__check_input__(y)
        orig_shape = y.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            return self.scaler.transform(y)
        elif y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            y = self.scaler.transform(y)
            return y.reshape(orig_shape)
        return self.scaler.transform(y)

    def inverse_transform(self, y):
        y = self.__check_input__(y)
        orig_shape = y.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            return self.scaler.inverse_transform(y)
        elif y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            y = self.scaler.inverse_transform(y)
            return y.reshape(orig_shape)
        return self.scaler.inverse_transform(y)

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        return y.astype(float)


class DataScalerMinMax:
    def __init__(self, y, config):
        self.config = config
        y = self.__check_input__(y)
        self.original_shape = None
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim > 2:
            self.original_shape = y.shape
            y = y.reshape(-1, y.shape[-1])
        train_data = y
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data)

    def transform(self, y):
        y = self.__check_input__(y)
        orig_shape = y.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            return self.scaler.transform(y)
        elif y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            y = self.scaler.transform(y)
            return y.reshape(orig_shape)
        else:
            return self.scaler.transform(y)

    def inverse_transform(self, y):
        y = self.__check_input__(y)
        orig_shape = y.shape
        if y.ndim == 1: 
            y = y.reshape(-1, 1)
            return self.scaler.inverse_transform(y)
        elif y.ndim > 2:
            y = y.reshape(-1, y.shape[-1])
            y = self.scaler.inverse_transform(y)
            return y.reshape(orig_shape)
        else:
            return self.scaler.inverse_transform(y)

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        return y.astype(float)


class GlobalStandardScaler:
    def __init__(self, y, config):
        self.config = config
        train_data = y[:int(len(y) * self.config.density)]
        train_data = self.__check_input__(train_data)
        self.mean = train_data.mean()
        self.std = train_data.std()
        if self.std == 0:
            self.std = 1

    def transform(self, x):
        x = self.__check_input__(x)
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        x = self.__check_input__(x)
        return x * self.std + self.mean

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy().astype(float)
        elif isinstance(y, np.ndarray):
            y = y.astype(float)
        return y
    
    

class GlobalMinMaxScaler:
    def __init__(self, y, config):
        self.config = config
        train_data = y
        train_data = self.__check_input__(train_data)

        self.min = train_data.min()
        self.max = train_data.max()

        # 防止除以0
        if self.max - self.min == 0:
            self.max += 1e-6

    def transform(self, x):
        x = self.__check_input__(x)
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x):
        x = self.__check_input__(x)
        return x * (self.max - self.min) + self.min

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy().astype(float)
        elif isinstance(y, np.ndarray):
            y = y.astype(float)
        return y


class NoneScaler:
    def __init__(self, y, config):
        self.config = config

    def transform(self, y):
        return y

    def inverse_transform(self, y):
        return y
    

def get_scaler(y, config, selected_method=None):
    method = selected_method if selected_method else config.scaler_method
    if method == 'stander':
        return DataScalerStander(y, config)
    elif method == 'minmax':
        return DataScalerMinMax(y, config)
    elif method == 'globalstander':
        return GlobalStandardScaler(y, config)
    elif method == 'globalminmax':
        return GlobalMinMaxScaler(y, config)
    elif method == 'None':
        return NoneScaler(y, config)
    else:
        raise NotImplementedError(f"Scaler method '{method}' is not supported.")