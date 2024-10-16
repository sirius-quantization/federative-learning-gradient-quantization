from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import ResNetModel
import pandas as pd
from typing import Literal, Union
from torch.utils.data import Dataset, DataLoader
import operator
import sys
from time import time
from tqdm import trange


class CustomResNet(nn.Module):
    def __init__(self, output_units, freeze_all = False, debug = False, resnet_version = 18):
        super(CustomResNet, self).__init__()
        if resnet_version == 18:
            self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        elif resnet_version == 50:
            self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        if freeze_all:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_units)
        if debug:
            for k, v in self.resnet.named_parameters():
                print(k, v.shape, v.requires_grad)

    def forward(self, x):
        return self.resnet(x)



class CIFARDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, idx):
        return transform(self.dset[idx]['img']), self.dset[idx]['label']

    def __len__(self):
        return len(self.dset)

    

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = load_dataset("uoft-cs/cifar10")
ds_train = ds["train"]

dataset = CIFARDataset(ds_train)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Row:
    model_name: str
    epochs: int
    batch: int
    dataset_name: str
    slaves_num: str
    transfer_quantity_GB: float
    transfer_summary_time_in_memory_sec: float
    accuracy: float
    loss: float
    unfreeze: Literal[Union["full", "last_layer"]]
    mean_learning_time_by_epoch_sec: float
    gradient_compression: Literal[Union["none", "scale"]]
    meta: str
    
    
    def __init__(self, 
                 model_name,
                 epochs,
                 batch,
                 dataset_name,
                 slaves_num,
                 transfer_quantity_GB,
                 transfer_summary_time_in_memory_sec,
                 accuracy,
                 loss,
                 unfreeze,
                 mean_learning_time_by_epoch_sec,
                 gradient_compression,
                 meta=None):
        self.model_name = model_name
        self.epochs = epochs
        self.batch = batch
        self.dataset_name = dataset_name
        self.slaves_num = slaves_num
        self.transfer_quantity_GB = transfer_quantity_GB
        self.transfer_summary_time_in_memory_sec = transfer_summary_time_in_memory_sec
        self.accuracy = accuracy
        self.loss = loss
        self.unfreeze = unfreeze
        self.mean_learning_time_by_epoch_sec = mean_learning_time_by_epoch_sec
        self.gradient_compression = gradient_compression
        self.meta = meta

        
def get_nested_attr(obj, attr_path):
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj

def set_nested_attr(obj, attr_path, new_value):
    parts = attr_path.split(".")
    for i, part in enumerate(parts[:-1]):
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_value)
    

total_bytes = 0
total_time_sending = 0
    
def run_exp(
    model_name,
    epochs,
    batch,
    dataset_name,
    slaves_num,
    unfreeze,
    gradient_compression,
):
    
    global total_bytes, total_time_sending
    total_bytes = 0
    total_time_sending = 0
    
    BATCH_SIZE = batch
    batch_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    EPOCH_COUNT = epochs
    SLAVE_COUNT = slaves_num
    FIXED_LR = 0.001
    OUT_UNITS = 10
    MASTER_NODE = 0

    criterion = nn.CrossEntropyLoss()
    master_model = CustomResNet(output_units=OUT_UNITS, resnet_version=model_name, freeze_all=unfreeze != "full").to(device)
    master_optimizer = optim.Adam(master_model.parameters(), lr=FIXED_LR)
    slave_models = [CustomResNet(output_units=OUT_UNITS, resnet_version=model_name, freeze_all=unfreeze != "full").to(device) for _ in range(SLAVE_COUNT)]

    
    
    def sync_slaves_with_master():
        global total_time_sending
        time_start_sending = time()
        with torch.no_grad():
            for models in zip(master_model.named_parameters(), *list(map(lambda x: x.named_parameters(), slave_models))):
                param_name = models[1][0]
                master_param = get_nested_attr(master_model, param_name)
                for model in slave_models:
                    new_param = nn.Parameter(master_param)
                    set_nested_attr(model, param_name, new_param)
        total_time_sending += time() - time_start_sending

        
    def quantize(gradients_raw, gradient_compression):
        if gradient_compression == "simple":
            return [torch.quantize_per_tensor(x.to("cpu"), 0.1, 10, torch.quint8) for x in gradients_raw]
        else:
            assert gradient_compression == "none"
            
            
    def dequantize(gradients_raw, gradient_compression):
        if gradient_compression == "simple":
            return [x.dequantize() for x in gradients_raw]
        else:
            assert gradient_compression == "none"

    def move_gradients_from_slaves_to_master():
        global total_bytes, total_time_sending
        time_start_sending = time()
        for models in zip(master_model.parameters(), *list(map(lambda x: x.parameters(), slave_models))):
            master_model_params = models[0]
            if not master_model_params.requires_grad:
                continue
            slave_models_params = models[1:]
            gradients_raw = list(map(lambda x: x.grad, slave_models_params))
            if None in gradients_raw:
                return
            # gradients_raw = quantize(gradients_raw, gradient_compression)
            total_bytes += torch.stack(gradients_raw).nelement() * torch.stack(gradients_raw).element_size()
            # gradients_raw = dequantize(gradients_raw, gradient_compression)
            gradient = torch.mean(torch.stack(gradients_raw), dim=0)
            print(gradient.dtype, gradient.shape)
            print(gradient)
            master_model_params.grad = gradient
        total_time_sending += time() - time_start_sending

    start_time = time()
    for epoch in trange(EPOCH_COUNT, desc="iterating through epochs"):
        
        " 0 - master, 1-N - slaves "
        executing_node = 1

        sync_slaves_with_master()
        move_gradients_from_slaves_to_master()

        index = 0
        while index < batch_loader.__len__():
            if executing_node == MASTER_NODE:
                move_gradients_from_slaves_to_master()
                master_optimizer.step()
                sync_slaves_with_master()
            else:
                inputs, labels = batch_loader.__iter__().__next__()
                index += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                model = slave_models[executing_node - 1]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
            executing_node = (executing_node + 1) % (SLAVE_COUNT + 1)

        dataset_test = CIFARDataset(ds["test"])
        test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
        accuracy_on_batch = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = master_model(inputs)
            accuracy_on_batch.append(sum(torch.argmax(outputs, dim=1) == labels) / len(labels))

    transfer_quantity_GB = round(total_bytes / (10 ** 9) / EPOCH_COUNT, 4)
    transfer_summary_time_in_memory_sec = total_time_sending
    accuracy = round(torch.mean(torch.tensor(accuracy_on_batch)).item(), 4)
    loss = round(loss.item(), 7)
    mean_learning_time_by_epoch_sec = round((time() - start_time)  / EPOCH_COUNT, 4)
        
    return transfer_quantity_GB, transfer_summary_time_in_memory_sec, accuracy, loss, mean_learning_time_by_epoch_sec
