# coding : utf-8
# Author : Yuxiang Zeng
import numpy as np
from data_dataset import TensorDataset
from modules.load_data.get_latency import get_latency
from utils.data_dataloader import get_dataloaders
from utils.data_scaler import get_scaler
from utils.data_spliter import get_split_dataset
from utils.exp_logger import Logger
from utils.exp_metrics_plotter import MetricsPlotter
from utils.utils import set_settings
from utils.exp_config import get_config


def load_data(config):
    all_x, all_y, scaler = get_latency(config)
    return all_x, all_y, scaler


# 数据集定义
class DataModule:
    def __init__(self, config, verbose=False):
        self.config = config
        self.path = config.path
        self.x, self.y, self.scaler = load_data(config)
        if config.debug:
            self.x, self.y = self.x[:300], self.y[:300]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = get_split_dataset(self.x, self.y, config)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, config)

        if verbose:
            config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)} Max_value : {np.max(self.y):.2f}')

    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, config):
        return (
            TensorDataset(train_x, train_y, 'train', config),
            TensorDataset(valid_x, valid_y, 'valid', config),
            TensorDataset(test_x, test_y, 'test', config)
        )


if __name__ == '__main__':
    config = get_config()
    set_settings(config)
    config.experiment = True

    # logger plotter
    exper_detail = f"Dataset : {config.dataset.upper()}, Model : {config.model}, Train_size : {config.train_size}"
    log_filename = f'{config.train_size}_r{config.rank}'
    log = Logger(log_filename, exper_detail, config)
    plotter = MetricsPlotter(log_filename, config)
    config.log = log
    log(str(config.__dict__))

    datamodule = DataModule(config)
    for train_batch in datamodule.train_loader:
        all_item = [item.to(config.device) for item in train_batch]
        inputs, label = all_item[:-1], all_item[-1]
        print(inputs, label.shape)
        # break
    print('Done!')
