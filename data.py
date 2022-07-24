
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision.datasets as Datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url

import os
import random
import numpy as np
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data_dir, name, transform, label):
        self.data_dir = data_dir
        self.total_data_dir = os.listdir(data_dir)
        self.transform = transform
        self.label = label
        self.name = name
    def __len__(self):
        return len(self.total_data_dir)

    def __getitem__(self, idx):
        loc = os.path.join(self.data_dir, self.total_data_dir[idx])
        rgb_img = Image.open(loc).convert('RGB')
        #rgb_img = rgb_img.resize([256, 256], Image.ANTIALIAS)
        tensor_image = self.transform(rgb_img)

        return (tensor_image, self.label, self.name)

def get_data(data_dir, img_size, batch_size):
    print("Load Data...")
    normal_ds_list = []
    anomaly_ds_list = []
    anomaly_test_list = []
    transform = T.Compose([T.Resize(img_size), T.ToTensor()])

    
    for key, value in data_dir["normal"].items():
        ds = MyDataset(value, key, transform=transform, label=0)
        normal_ds_list.append(ds) 
    for key, value in data_dir["anomaly"].items():
        ds = MyDataset(value, key, transform=transform, label=1)
        anomaly_ds_list.append(ds)
    if data_dir["anomaly_test"] is not None:   
        for key, value in data_dir["anomaly_test"].items:
            ds = MyDataset(value, key, transform=transform, label=1)
            anomaly_test_list.append(ds)
    
    normal_dataset = ConcatDataset(normal_ds_list)
    anomaly_dataset = ConcatDataset(anomaly_ds_list)

    train_size = int(len(normal_dataset)*0.7)
    val_size = int(len(normal_dataset)*0.2)
    test_size = len(normal_dataset) - train_size - val_size
    train_normal, val_normal, test_normal = random_split(normal_dataset, [train_size, val_size, test_size])
    if data_dir["anomaly_test"] is None:
        train_size = int(len(anomaly_dataset)*0.7)
        val_size = int(len(anomaly_dataset)*0.2)
        test_size = len(anomaly_dataset) - train_size - val_size
        train_anomaly, val_anomaly, test_anomaly = random_split(anomaly_dataset, [train_size, val_size, test_size])
    else:
        train_size = int(len(anomaly_dataset)*0.75)
        val_size = len(anomaly_dataset) - train_size
        train_anomaly, val_anomaly = random_split(anomaly_dataset, [train_size, val_size])
        test_anomaly = ConcatDataset(anomaly_test_list)

    train_dataset = ConcatDataset([train_normal, train_anomaly])
    val_dataset = ConcatDataset([val_normal, val_anomaly])
    test_dataset = ConcatDataset([test_normal, test_anomaly])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print("Loading Data done")
    return train_loader, val_loader, test_loader


def get_data_STL10(batch_size, root):
    print("Loading trainset...")
    transform=T.Compose([T.Resize(256), T.ToTensor])
    trainset = MyDataset(root, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print("Loading testset...")
    testset = MyDataset(root, transform=transform)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("Done!")

    return trainloader, testloader