import models, utils
from data import vocalSetDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random, os, pdb, time, tqdm


def train(args, epoch, myModel):

	
	train_loss_history=history_list[0]
	train_acc_hist=history_list[1]
	val_loss_history=history_list[2]
	val_acc_hist=history_list[3]


	myModel.train()
	train_loss = 0
	total = 0
	correct = 0
	for batch_num, (x_data, y_data)  in enumerate(train_loader):
		# sends tensors to gpu so they become gpu managed tensors
		x_data = x_data.to(device, dtype=torch.float)
		y_data = y_data.to(device)
		# reset gradient history
		optimizer.zero_grad()
		# pdb.set_trace()
		# insert channel param in right place
		# x_data=x_data.view(args.batch_size, image_height, image_width)
		# MIGHT NEED TO PUT IN A HEIGHT DIMINISHER HERE
		# call forward functions of model
		prediction = myModel(x_data)
		# call loss function
		loss = models.my_loss_function(prediction, y_data)
		# get accuracy rating
		_, predicted = torch.max(prediction.data, 1)
		# pdb.set_trace()
		total += y_data.shape[0]
		correct += (predicted == y_data).sum().item()
		# call backprop based on loss
		loss.backward()
		# get scalar value from loss tensor and add to train_loss for end_of_epoch report
		train_loss += loss.item()
		# update the optimiser after cnosidering the loss
		optimizer.step()
		# report after every X batches
		if batch_num % 10 == 0:
			#pdb.set_trace()

			print('Train Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
				epoch, batch_num * args.batch_size, len(train_loader.dataset),
				# batch num / total batches available in percentage form (this is correct)
				100. * batch_num / len(train_loader),
				loss.item() / args.batch_size))
	print('====> Train Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))
	train_loss_history.append(train_loss / len(train_loader.dataset))
	train_acc_hist.append(correct / total)

def val(args, epoch, myModel):
	# sets up model to evaluation mode 
	myModel.eval()
	test_loss = 0
	total = 0
	correct = 0
	# do not update torch gradient history for this part
	with torch.no_grad():
		for batch_num, (x_data, y_data)  in enumerate(val_loader):
			# sends tensors to gpu so they become gpu managed tensors
			x_data = x_data.to(device, dtype=torch.float)
			y_data = y_data.to(device)
			# insert channel param in right place
			# x_data=x_data.unsqueeze(1)
			# x_data=x_data.view(args.batch_size, image_height, image_width)
			# no need to use optimizer.zero_grad( here)
			# call forward functions of model
			prediction = myModel(x_data)
			# call loss function
			loss = models.my_loss_function(prediction, y_data)
			# get accuracy rating
			_, predicted = torch.max(prediction.data, 1)
			total += y_data.shape[0]
			correct += (predicted == y_data).sum().item()
			# accumulate loss for report
			test_loss += loss.item()
			# report after every X biatches
			if batch_num % 10 == 0:
				print('Test Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
					epoch, batch_num * args.batch_size, len(val_loader.dataset),
					100. * batch_num / len(val_loader),
					loss.item() / args.batch_size))
		print('====> Test Epoch: {} Average loss: {:.4f}'.format(epoch, test_loss / len(val_loader.dataset)))
		val_loss_history.append(test_loss / len(val_loader.dataset))
		val_acc_hist.append(correct / total)
