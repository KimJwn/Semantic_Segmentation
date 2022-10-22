# JW 
#   Dragonball
#       main.py

import argparse
import logging
import os
from re import M
# from sre_parse import _OpGroupRefExistsType
from unittest.mock import patch
from data.Dataset import CrackDataset
import torch
from utils.train_utils import *
from model.unet import UNet16


# import torch.distributed as d

# d.all_reduce_multigpu


save_dir = '/home/jovyan/G1/dragon_ball/utils'

def parse_args():
	parser = argparse.ArgumentParser(description='CRACK SEGMENTATION')
	parser.add_argument('--batch_size', '-b', type=int, default=4)
	parser.add_argument('--train', '-t', default=True)
	parser.add_argument('--project_name', '-n', default='crack_seg')
	parser.add_argument('--seed', type=int, default=11)
	parser.add_argument('—-save_model', '-s', action='store_true', default=False)

	return parser.parse_args()



def check_mkdir(dir_name):
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)


def train(train_loader, model, optimizer, criterion):
	metrics = 0
	model.train()
	
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
	#
	return metrics, loss

'''
def validation(model, 
			   val_loader, 
			   logger, ):
	model.eval()
'''

def test(test_loader, model, criterion, logger):
	metrics = 0
	model.eval()

	with torch.no_grad():
		for batch_idx, (image, label) in enumerate(test_loader):
			# Forward 
			outputs = model(image) 
			# loss_func 
			loss = criterion(outputs, label)
			# metrics	#
			#			#
	return metrics, loss


def main():
	args = parse_args()

	#
	lr = 0.001
	momentum = 0.9
	weight_decay = 1e-4
	batch_size = 4
	num_workers = 4
	epoches = 10 # 50
	batch_size, num_workers = 16,4
	
	torch.manual_seed(args.seed)

	#
	print(torch.__version__, torch.cuda.__path__)	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device_id = int(os.environ["LOCAL_RANK"])

	path = create_save_dir(args.project_name)

	logger = get_log(path)
 
	#
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("gpu") if torch.cuda.is_available() else print("cpu")

	#
	train_loader = CrackDataset(split='train')
	test_loader = CrackDataset(split='test')

	#
	model = UNet16(pretrained=False)

	#
	optimizer = torch.optim.SGD(model.parameters(), lr,
									momentum=momentum,
									weight_decay=weight_decay)

	#
	criterion = torch.nn.BCEWithLogitsLoss().to('cuda')

	if args.save_model:
		torch.save(model.state_dict())

	for epoch in range(epoches):
		# train
		metrics, loss = train(train_loader, model, optimizer, criterion)

		#test
		test_metrics, test_loss = test(test_loader, model, criterion)
	
		
	
if __name__ == "__main__":
	main()