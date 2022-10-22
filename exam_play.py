
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms


from unet.exam_model import UNet16
from exam_dataset import ImgDataSet


import warnings
warnings.filterwarnings('ignore')

'''
def train(train_loader, model, criterion, optimizer, validation, args):

    latest_model_path = find_latest_model_path(args.model_dir)

    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        epoch = epoch

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 0
        min_val_los = 9999

    valid_losses = []
    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()

        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()

            masks_pred = model(input_var)

            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            losses.update(loss)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg
        }, epoch_model_path)

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, best_model_path)
'''


#
lr = 0.001
momentum = 0.9
weight_decay = 1e-4
batch_size = 4
num_workers = 4
epoch = 10 # 50
batch_size, num_workers = 16,4


#
DIR_IMG  = '/home/jovyan/Datasets/crack_segmentation_dataset/images/'
DIR_MASK = '/home/jovyan/Datasets/crack_segmentation_dataset/masks/'
img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]


#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#
model = UNet16(pretrained=True)

#
optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

#
criterion = nn.BCEWithLogitsLoss().to('cuda')


#
channel_means = [0.485, 0.456, 0.406]
channel_stds  = [0.229, 0.224, 0.225]

#
train_tfms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(channel_means, channel_stds)])

mask_tfms = transforms.Compose([transforms.ToTensor()])

#
dataset = ImgDataSet(img_transform=train_tfms, mask_transform=mask_tfms)


#    
train_size = int(0.8*len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=num_workers)


#
model.cuda()

#
import matplotlib.pyplot as plt
import torch.functional as F
network = UNet16()


for batch_idx, (image, label) in enumerate(train_loader):
		optimizer.zero_grad()
		# Forward 
		output = model(image)
		# loss_func
		loss = criterion(output, label)
		# metrics	#
		#			#

		# Gradinet 
		loss.backward()   
		# weight update 
		optimizer.step() 


# #
# for (X_train, y_train) in train_loader:
#     print((X_train.shape))
#     # plt.imshow(X_train[0].permute(1,2,0).numpy())
#     # plt.show()
#     X_train, labels = X_train.to(device), y_train.to(device)
#     output = model(X_train)
#     # plt.imshow(nn.Sigmoid(output[0]).permute(1,2,0).detach().numpy())
#     # plt.show()
    
#     break    

