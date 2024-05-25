import os 
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets as vision_datasets
from datasets.base_dataset import ImageMaskDataset, ImageMaskWeightDataset
import yaml

def get_data(config):
    data_name = config["data_name"]
    is_train = config["is_train"]
    # is_png = True if data_name in ["HKU-IS", "MSOD"] else False # images in [HKU-IS, MSOD] are ended with .png

    # load dataset config
    with open(config["dataset_config"], 'r') as f:
        dataset_config = yaml.load(f, Loader=yaml.Loader)
    data_dir = dataset_config[data_name]["root_dir"]

    if is_train:
        # load train_dataset_config
        # typically use DUTS as the training dataset (DUTS-TR for training, DUTS-TE for testing)
        assert data_name == "DUTS", 'set data_name to DUTS!'
        train_dir = os.path.join(dataset_config[data_name]["root_dir"], dataset_config[data_name]["train_dir"])
        test_dir = os.path.join(dataset_config[data_name]["root_dir"], dataset_config[data_name]["test_dir"])
        train_dataset = ImageMaskWeightDataset(data_dir=train_dir, transform=True, load_weight=True, is_train=True)
        test_dataset = ImageMaskWeightDataset(data_dir=test_dir, transform=True, load_weight=True, is_train=False)
    else:
        train_dataset = None
        test_dataset = ImageMaskWeightDataset(data_dir=data_dir, transform=True, load_weight=True, is_train=False)
    return train_dataset, test_dataset


def get_dataloader(dataset, config, batch_size=16, shuffle=False, num_workers=4, pin_memory=False, drop_last=False, distributed=False, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], pin_memory=config["pin_memory"], drop_last=config["drop_last"], num_workers=config["num_workers"])
    return dataloader


