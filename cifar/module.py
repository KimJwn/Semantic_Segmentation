import torchvision 
from torchvision.datasets import CIFAR10
from torchvision import transforms

from gc import callbacks

import torch
from torch.utils.data import random_split, DataLoader
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR

import pytorch_lightning as pl

import torchmetrics

class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir='/home/jovyan/Desktop/data', batch_size=256, num_workers=8):
        super().__init__()
        device = torch.device('cuda')
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        norm_v = (0.5, 0.5, 0.5)
        self.transform = transforms.Compose( [
            transforms.ToTensor(),
            transforms.Normalize(norm_v, norm_v) ] )

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    #

    def setup(self, stage="fit"):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            cifar_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_train, [45000, 5000])

        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    #

    def train_dataloader(self):
        '''returns training dataloader'''
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return cifar_train
    #

    def val_dataloader(self):
        '''returns validation dataloader'''
        cifar_val = DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers)
        return cifar_val
    #

    def test_dataloader(self):
        '''returns test dataloader'''
        cifar_test = DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers)
        return cifar_test
    #


class LitCF(pl.LightningModule):

    def __init__(self, n_classes=10, lr=0.001):
        super().__init__()
        device = torch.device('cuda')
        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

        # lenet
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # optimizer parameters
        self.lr = lr

        # metrics
        self.accuracy = torchmetrics.Accuracy()#pl.metrics.Accuracy()

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)

        # Log training loss, metrics
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(output, y))

        return loss


    def evaluate(self, batch, stage="eval"):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        '''defines model optimizer'''
        return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)



class ResNet(pl.LightningModule):
    def __init__(self, block, num_blocks, num_classes=10, lr=0.05):
        super(ResNet, self).__init__()
        device = torch.device('cuda')
        self.save_hyperparameters()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
