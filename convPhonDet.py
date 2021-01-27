
from __future__ import print_function
import argparse, torch, torch.utils.data, pdb, time, h5py, random, os
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pdb, time, h5py, random, os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

seconds=time.time()

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--h5', type=str, metavar='N',
																				help='MUST include a h5 file, including file extension')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
																				help='input batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
																				help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
																				help='enables CUDA training')
parser.add_argument('--iter-data', type=int, default=2, metavar='N',
																				help='how often to iterate over training data to increase size')
parser.add_argument('--extra-notes', type=str, default='NoExtraNotes', metavar='N',
																				help='extra notes to help describe config. Use _ for spaces')
parser.add_argument('--window-size', type=int, default=23, metavar='N',
																				help='wnodw_size is 23 if not specified')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(1)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# spec resolution is usually 23ms, therefore 23*window_length = window duration (ms)
report_frequency = 10
window_length = args.window_size
assumed_spectrogram_timestep=23 #ms
oneSpecSecond=int(1000/assumed_spectrogram_timestep)
train_split = 0.8
h5_basename = os.path.basename(args.h5)
string_log = h5_basename +'_' +str(args.epochs) +'Epo_' +str(args.iter_data) +'Iterd_' +str(args.batch_size) + 'BatchS_' +str(args.window_size) +'WindowS_' +args.extra_notes
print('string log: ', string_log)
os.makedirs('results/' +string_log, exist_ok=True)

class vocalSetDataset(Dataset):
	def __init__(self):
		data = h5py.File(args.h5, 'r')
		extra_data, features, keys, lengths, technique = data.keys()
		self.x = data[features]
		self.y = data[technique]
		self.x = np.asarray(self.x)
		self.y = np.asarray(self.y)
		self.y = self.y.squeeze()
		self.lengths = data[lengths]
		self.n_samples = self.x.shape[0]

	def __getitem__(self, index):
		# window length must be odd, half_window will give length on either side
		half_window=int(window_length/2)
		features, label, frame_num = self.x[index], self.y[index], self.lengths[index]
		# random_frame is limited to choose a window that does not overlap with the first or last second of the audio clip, to avoid using 'silence' 
		random_frame = random.randint(half_window+oneSpecSecond, frame_num-window_length-oneSpecSecond)
		# subarray include the starting point up to (but not including) the end point. Therefore we add 1
		sample_excerpt = features[:,random_frame-half_window:random_frame+half_window+1]
		sample_excerpt = torch.from_numpy(sample_excerpt)
		# sample_excerpt = sample_excerpt.unsqueeze(1)
		# print('type and shape',type(sample_excerpt), sample_excerpt.shape)
		return sample_excerpt, label

	def __len__(self):
		# len(dataset)
		return self.n_samples

dataset = vocalSetDataset()
train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# batch_size tells DataLoader how many steps or passes it should take per epoch
# train_loader will be of size train_dataset/batch_size
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
total_samples = len(dataset)
n_iterations = total_samples/args.batch_size
conv2dense_channel = 8
data = h5py.File(args.h5, 'r')
extra_data, features, keys, lengths, technique = data.keys()
image_height = data[features][0].shape[0]
image_width = window_length
conv2denseDim = conv2dense_channel*image_height*image_width
h_w_dim = image_height*image_width
class_num = 5

class PhonationDetector(nn.Module):
	def __init__(self):
		super(PhonationDetector, self).__init__()
		kernelSize=5
		paddingSize=int((kernelSize-1)/2)
		self.conv1 = nn.Conv2d(1, 2, kernelSize, padding=paddingSize)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(2, 4, kernelSize, padding=paddingSize)
		self.conv3 = nn.Conv2d(4, 8, kernelSize, padding=paddingSize)
		self.fc1 = nn.Linear(8*h_w_dim, h_w_dim)
		self.fc2 = nn.Linear(h_w_dim, int(h_w_dim/2))
		self.fc3 = nn.Linear(int(h_w_dim/2), class_num) 

	def forward(self, x):
		# print('x shape:', x.shape)
		# x = self.pool(F.relu(self.conv1(x)))
		x = F.relu(self.conv1(x))
		# print('x shape:', x.shape)
		x = F.relu(self.conv2(x))
		# print('x shape:', x.shape)	
		x = F.relu(self.conv3(x))
		# print('x shape:', x.shape)	
		x = x.view(-1, 8*h_w_dim)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

myModel = PhonationDetector().to(device)
optimizer = optim.Adam(myModel.parameters(), lr=1e-4)

def my_loss_function(prediction, target):
	loss = F.cross_entropy(prediction, target)
	return loss

def train(epoch):
	myModel.train()
	train_loss = 0
	for batch_num, (x_data, y_data)  in enumerate(train_loader):
		# sends tensors to gpu so they become gpu managed tensors
		x_data = x_data.to(device, dtype=torch.float)
		y_data = y_data.to(device)
		# reset gradient history
		optimizer.zero_grad()
		# insert channel param in right place
		x_data=x_data.unsqueeze(1)
		# call forward functions of model
		prediction = myModel(x_data)
		# call loss function
		loss = my_loss_function(prediction, y_data)
		# call backprop based on loss
		loss.backward()
		# get scalar value from loss tensor and add to train_loss for end_of_epoch report
		train_loss += loss.item()
		# update the optimiser after cnosidering the loss
		optimizer.step()
		# report after every X batches
		if batch_num % report_frequency == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
				epoch, batch_num * len(data), len(train_loader.dataset),
				100. * batch_num / len(train_loader),
				loss.item() / len(data)))	
	print('====> Train Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))

def val(epoch):
	# sets up model to evaluation mode 
	myModel.eval()
	test_loss = 0
	# do not update torch gradient history for this part
	with torch.no_grad():
		for batch_num, (x_data, y_data)  in enumerate(val_loader):
			# sends tensors to gpu so they become gpu managed tensors
			x_data = x_data.to(device, dtype=torch.float)
			y_data = y_data.to(device)
			# insert channel param in right place
			x_data=x_data.unsqueeze(1)
			# no need to use optimizer.zero_grad( here)
			# call forward functions of model
			prediction = myModel(x_data)
			# call loss function
			loss = my_loss_function(prediction, y_data)
			# accumulate loss for report
			test_loss += loss.item()
			# report after every X biatches
			if batch_num % report_frequency == 0:
				print('Test Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
					epoch, batch_num * len(data), len(val_loader.dataset),
					100. * batch_num / len(val_loader),
					loss.item() / len(data)))
		print('====> Test Epoch: {} Average loss: {:.4f}'.format(epoch, test_loss / len(val_loader.dataset)))

for epoch in range(1, args.epochs+1):
	train(epoch)
	val(epoch)
	# do not apply gradient history to this section of the code

# be sure to set model to eval() mode to configure dropout, batchnorm layers not to affect inference
torch.save(myModel.state_dict(), 'myFirstModel.pt')
