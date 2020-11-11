#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pathlib
import glob2
import gc
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

from tqdm import tqdm
import collections
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True

from skimage import io

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensor

import timm
from timm.models.layers import SelectAdaptivePool2d

@contextmanager
def timer(msg: str):
    t0 = time.time()
    try:
        yield
    except RuntimeError as e:
        print(e)
        sys.exit(-1)
    s = f"[{msg}] done in {time.time() - t0:.0f} s"
    print(s)

def cuda_vars_free():
    resource = locals().items()
    for name, value in resource:
        if hasattr(value, "device"):
            if value.device.type == "cuda":
                print(name, sys.getsizeof(value.storage()), end=" ")
                del globals()[name]
                print("removed")
    gc.collect()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

@torch.no_grad()
def get_loss(model, loader, criterion):
    loss = 0.0
    num = 0
    model.to('cuda')
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model(inputs)
        loss += criterion(output, labels).item()
        num += labels.shape[0]
    return loss / num

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    model.to('cuda')
    for inputs, labels in tqdm(loader):
        inputs = inputs.to('cuda')
        preds = model(inputs)
        preds = preds.to('cpu')
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds

def get_num_correct(preds, labels):
    return preds.eq(labels).sum().item()

def plot_confusion_matrix(cm, classes, filename, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    plt.ioff()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)

import argparse

ap = argparse.ArgumentParser(description="test_diamond")

ap.add_argument("--model", default="tv_resnet101")

ap.add_argument("--img-size", type=int, default=256)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--epoch-warmup", type=int, default=10)
ap.add_argument("--epoch-uptrain", type=int, default=50)
ap.add_argument("--grad-accum", type=int, default=1)

ap.add_argument("--weighted", action='store_true', help="Use class-weighted errors")
ap.add_argument("--debug", action='store_true', help="Debug mode")
ap.add_argument("--tta", action='store_true', help="Use test-time augmentation")
ap.add_argument("--dev-id", type=int, default=0, help="Specify the cuda device id")

args = vars(ap.parse_args())

print("Settings:")
for k, v in vars(ap.parse_args()).items():
    print(" " * 4, k, v)

img_size = args["img_size"]
grad_accum = args["grad_accum"]
bsize = args["batch_size"]
model_name = args["model"]

epochs_warmup = args["epoch_warmup"]
epochs_uptrain = args["epoch_uptrain"]

device = 'cuda'
dev_id = args["dev_id"]

is_weighted = args["weighted"]
debug = args["debug"]
enable_tta = args["tta"]

start = time.time()
JST = timezone(timedelta(hours=+9), "JST")
print("=== Start", datetime.now(JST).strftime("%Y.%m.%d %H:%M:%S"))

tb_dir = "runs"
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
writer = SummaryWriter(os.path.join(tb_dir, "tbout"))

jpg_paths = glob2.glob("input/**/*.jpg")
jpg_paths = [ p for p in jpg_paths if ".ipynb_checkpoints" not in p]

xls = pd.ExcelFile('input/2019_datalist.xlsx')

clarity = {}
for tab in xls.sheet_names:
    for i, row in pd.read_excel(xls, tab).iterrows():
        clarity[row['ソーティングNo']] = row['クラリティー']

labels = []
for path in jpg_paths:
    pfile = pathlib.Path(path)
    if pfile.stem in clarity:
        labels.append(clarity[pfile.stem])
    else:
        print(pfile)

classes = ['VVS1','VVS2', 'VS-1', 'VS-2', 'SI-1', 'SI-2', 'I-1', 'I-2', 'I-3']
clarity_class = {k: i for i,k in enumerate(classes)}
label_integers = [clarity_class[l] for l in labels]

assert set(classes) == set(clarity.values())

assert len(labels) == len(jpg_paths)

if not torch.cuda.is_available():
    print("GPU not available")
    sys.exit(-1)

for id in range(torch.cuda.device_count()):
    print("device", id, ":", torch.cuda.get_device_name(id)) 

torch.cuda.set_device(dev_id)
dev_id = torch.cuda.current_device()
print(f"cuda: device {dev_id} is used.")

mem = torch.cuda.get_device_properties(dev_id).total_memory
print(f'Total Mem: {mem/1024**3:3.1f}G')

mem = torch.cuda.memory_reserved(dev_id)
print(f'Reserved Mem: {mem/1024**3:3.1f}G')

mem = torch.cuda.memory_allocated(dev_id)
print(f'Allocated Mem: {mem/1024**3:3.1f}G')

class CustomModel(nn.Module):
    def __init__(self, timm_model, gpool, head):
        super(CustomModel, self).__init__()
        self.backbone = timm_model
        self.gpool = gpool
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.forward_features = self.backbone.forward_features
        self.head = head

    def forward(self, x):
        f = self.forward_features(x)
        if hasattr(self.backbone, "global_pool"):
            if self.gpool:
                f = self.gpool(f)
            else:
                f = self.backbone.global_pool(f)
        y = self.head(f)
        return y


def create_net(model_name, num_class, head="bestfitting", concat_pool=False):
    model_timm = timm.create_model(model_name, pretrained=True)
    num_ftrs = model_timm.num_features
    if concat_pool:
        neck = SelectAdaptivePool2d(output_size=1, pool_type="catavgmax", flatten=True)
        num_ftrs *= 2
    else:
        neck = SelectAdaptivePool2d(output_size=1, pool_type="avg", flatten=True)
    if head == "bestfitting":
        clf = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, int(num_ftrs / 2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(num_ftrs / 2)),
            nn.Dropout(p=0.5),
            nn.Linear(int(num_ftrs / 2), num_class)
        )
    elif head == "bn_linear":
        clf = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_class)
        )
    else:
        clf = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_class)
        )
    model = CustomModel(model_timm, neck, clf)

    return model

class MyDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.transform = transform
        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        x = io.imread(img_path)
        if self.transform:
            augmented = self.transform(image=x)
            x = augmented["image"]

        return x, label

n = len(jpg_paths)
print("# images:", n)

train_idx, val_idx = train_test_split(range(n), test_size=0.33)
train_imgs   = [jpg_paths[i] for i in train_idx]
train_labels = [label_integers[i] for i in train_idx]
val_imgs     = [jpg_paths[i] for i in val_idx]
val_labels   = [label_integers[i] for i in val_idx]

print("train:", len(train_idx))
count = collections.Counter(train_labels)
for cls in classes:
    print(' '*4, f'{cls}: {count[clarity_class[cls]]}')

print("test:", len(val_idx))
count = collections.Counter(val_labels)
for cls in classes:
    print(' '*4, f'{cls}: {count[clarity_class[cls]]}')

transform = A.Compose([
    A.CenterCrop(height=1200, width=1200, p=1.0),
    A.Resize(always_apply=False, p=1, height=img_size, width=img_size, interpolation=1),
    A.CLAHE(always_apply=True, p=1.0, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
    A.RandomRotate90(always_apply=False, p=0.5),
    A.HorizontalFlip(always_apply=False, p=0.5),
    A.VerticalFlip(always_apply=False, p=0.5),
    A.Normalize(always_apply=False, p=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
    ToTensor()
])

from torch.utils.data import DataLoader

if debug:
    k = 500
    train_imgs, train_labels = train_imgs[:k], train_labels[:k]
    val_imgs, val_labels = val_imgs[:k], val_labels[:k] 
    epochs_warmup = 2
    epochs_uptrain = 2

trainset = MyDataset(train_imgs, train_labels, transform)
train_loader = DataLoader(trainset, batch_size=bsize, drop_last=False, shuffle=True, pin_memory=True)
if len(trainset) % bsize < 2:
    train_loader = DataLoader(trainset, batch_size=bsize, drop_last=True, shuffle=True, pin_memory=True)

testset = MyDataset(val_imgs, val_labels, transform)
test_loader = DataLoader(testset, batch_size=bsize, drop_last=False, shuffle=False, pin_memory=True)

# training

print("=== Training Start", datetime.now(JST).strftime("%Y.%m.%d %H:%M:%S"))

if is_weighted:
    c_weight = {
            'VVS1': 100.0, 'VVS2': 100.0,
            'VS-1': 50.0, 'VS-2': 50.0, 
            'SI-1': 2.0, 'SI-2': 2.0, 
            'I-1': 1.0, 'I-2': 1.0, 'I-3': 1.0
            }
    assert set(classes) == set(c_weight.keys())
    print('Class Weight')
    for i, k in enumerate(classes):
        print(' '*4, f'{i}: {k} => {c_weight[k]}')
    class_weight = torch.tensor([c_weight[k] for k in classes]).to(device)
else:
    class_weight = torch.tensor([1.0 for _ in classes]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weight)

model = create_net(model_name, len(clarity_class), head="bn_linear", concat_pool=False)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-5)

model.to(device)

with timer("warmup"):
    freq = int(len(train_loader)/10) + 1
    print('freq', freq)
    for epoch in range(epochs_warmup):
        # train
        losses = AverageMeter('Loss', ':1.5f')
        model.train()
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            loss.backward()
            if (i+1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (i+1) % freq == 0:
                print(f'loss (train): {losses.avg: 1.5f}')
        # eval
        model.eval()
        val_loss = get_loss(model, test_loader, criterion)

        print(f'loss (eval): {val_loss: 1.5f}')
        writer.add_scalar("loss", losses.avg, epoch + 1)
        writer.add_scalar("val_loss", val_loss, epoch + 1)

        scheduler.step(val_loss)

torch.save(model.state_dict(), f"/tmp/torch_test.model")

torch.cuda.empty_cache()
cuda_vars_free()

model = create_net(model_name, len(clarity_class), head="bn_linear", concat_pool=False)
model.load_state_dict(torch.load(f"/tmp/torch_test.model"))

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-7)

model.to(device)

with timer("uptrain"):
    freq = int(len(train_loader)/10) + 1
    print('freq', freq)
    for epoch in range(epochs_uptrain):
        # train
        losses = AverageMeter('Loss', ':.4e')
        model.train()
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            loss.backward()
            if (i+1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (i+1) % freq == 0:
                print(f'loss (train): {losses.avg: 1.5f}')
        # eval
        model.eval()
        val_loss = get_loss(model, test_loader, criterion)

        print(f'loss (eval): {val_loss: 1.5f}')
        writer.add_scalar("loss", losses.avg, epoch + 1)
        writer.add_scalar("val_loss", val_loss, epoch + 1)

        scheduler.step(val_loss)

torch.save(model.state_dict(), f"/tmp/torch_test.model")

# eval
print("=== Prediction Start", datetime.now(JST).strftime("%Y.%m.%d %H:%M:%S"))

model = create_net(model_name, len(clarity_class), head="bn_linear", concat_pool=False)
model.load_state_dict(torch.load(f"/tmp/torch_test.model"))
model.eval()

preproc = A.Compose([
    A.CenterCrop(height=1200, width=1200, p=1.0),
    A.Resize(always_apply=False, p=1, height=img_size, width=img_size, interpolation=1),
    A.CLAHE(always_apply=True, p=1.0, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
    A.Normalize(always_apply=False, p=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
    ToTensor()
])

# training accuracy
data = MyDataset(train_imgs, train_labels, transform=preproc)
loader = DataLoader(data, batch_size=bsize, shuffle=False, pin_memory=True)
with torch.no_grad():
   train_preds = get_all_preds(model, loader)

y_train = torch.tensor(data.labels)
y_preds = train_preds.argmax(dim=1)
preds_correct = get_num_correct(y_preds, y_train)

print(f'correct: {preds_correct}, total: {len(data)}')
print(f'accuracy (train): {preds_correct / len(data):3.1%}')

cls_int = [i for i, _ in enumerate(classes)]
cm = confusion_matrix(y_train, y_preds, labels=cls_int)
plot_confusion_matrix(cm, classes, "cm_train.png")
print(cm)

# test accuracy
data = MyDataset(val_imgs, val_labels, transform=preproc)
loader = DataLoader(data, batch_size=bsize, shuffle=False, pin_memory=True)
with torch.no_grad():
   test_preds = get_all_preds(model, loader)

y_test = torch.tensor(data.labels)
y_preds = test_preds.argmax(dim=1)
preds_correct = get_num_correct(y_preds, y_test)

print(f'correct: {preds_correct}, total: {len(data)}')
print(f'accuracy (test): {preds_correct / len(data):3.1%}')

cls_int = [i for i, _ in enumerate(classes)]
cm = confusion_matrix(y_test, y_preds, labels=cls_int)
plot_confusion_matrix(cm, classes, "cm_test.png")
print(cm)

if enable_tta:
    print("=== TTA Start", datetime.now(JST).strftime("%Y.%m.%d %H:%M:%S"))
    num_tta = 10

    model.to(device)
    model.eval()

    correct, total = 0, 0
    trainset = MyDataset(train_imgs, train_labels, transform=None)
    with torch.no_grad():
        tta_preds = torch.tensor([])
        for img, label in trainset:
            inputs = torch.stack([transform(image=img)["image"] for i in range(num_tta)])
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = F.softmax(outputs, dim=1).sum(dim=0).to('cpu')
            correct += (label == pred.argmax()).sum().item()
            total += 1

    print(f'Train Set Accuracy (TTA): {correct / total:3.1%}')

    correct, total = 0, 0
    testset = MyDataset(val_imgs, val_labels, transform=None)
    with torch.no_grad():
        tta_preds = torch.tensor([])
        for img, label in tqdm(testset):
            inputs = torch.stack([transform(image=img)["image"] for i in range(num_tta)])
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = F.softmax(outputs, dim=1).sum(dim=0).to('cpu')
            correct += (label == pred.argmax()).sum().item()
            total += 1

    print(f'Test Set Accuracy (TTA): {correct / total:3.1%}')

mem = torch.cuda.get_device_properties(dev_id).total_memory
print(f'Total Mem: {mem/1024**3:3.1f}G')

mem = torch.cuda.memory_reserved(dev_id)
print(f'Reserved Mem: {mem/1024**3:3.1f}G')

mem = torch.cuda.memory_allocated(dev_id)
print(f'Allocated Mem: {mem/1024**3:3.1f}G')

print("=== End", datetime.now(JST).strftime("%Y.%m.%d %H:%M:%S"))

elapsed = time.time() - start
print("=== Elapsed Time ", timedelta(seconds=elapsed))

