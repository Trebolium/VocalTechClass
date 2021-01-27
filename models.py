from torch.nn import Linear, BatchNorm1d, Parameter
import torch, math, pdb
import torch.nn as nn
import torch.nn.functional as F

#class Flatten(nn.Module):
#    def forward(self, input):
#        return input.view(input.size(0), -1)

class Luo2019AsIs(nn.Module):
    def __init__(self, config, spmel_params):
        super().__init__()
        self.kernelSize = 3
        self.paddingSize = int(math.ceil((self.kernelSize-1)/2))
        self.initial_channels = config.n_mels
        self.inc1Dim = 512
        self.num_classes=6 # number of classes
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second)

        self.conv_layer1 = nn.Sequential(
                nn.Conv1d(self.initial_channels,
                            self.inc1Dim,
                            kernel_size = self.kernelSize, padding = self.paddingSize),
                nn.BatchNorm1d(self.inc1Dim),
                nn.ReLU()
            )
        self.conv_layer2 = nn.Sequential(
                nn.Conv1d(self.inc1Dim,
                            self.inc1Dim,
                            kernel_size = self.kernelSize, padding = self.paddingSize),
                nn.BatchNorm1d(self.inc1Dim),
                nn.ReLU()
            )

        """ Alternative CNN layers"""

#        self.conv_layer1 = nn.Sequential(
#                nn.Conv2d(1,
#                           128,
#                            kernel_size = self.kernelSize, padding = self.paddingSize),
#                nn.BatchNorm1d(128),
#                nn.ReLU()
#            )
#        self.pool1 = nn.MaxPool2d(2,2)
#        self.conv_layer2 = nn.Sequential(
#                nn.Conv1d(128,
#                            512,
#                            kernel_size = self.kernelSize, padding = self.paddingSize),
#                nn.BatchNorm1d(self.inc1Dim),
#                nn.ReLU()
#            )
#        self.pool2 = nn.MaxPool2d(2,2)
#        self.flat_size = self.window_size/4 * self.inc1Dim
        """ Alternative CNN layers"""

        self.flat_size = self.window_size * self.inc1Dim
        
        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.flat_size,
                        self.inc1Dim),
            nn.BatchNorm1d(self.inc1Dim),
            nn.ReLU()
            )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(self.inc1Dim,
                        int(self.inc1Dim/2)),
            nn.BatchNorm1d(int(self.inc1Dim/2)),
            nn.ReLU()
            )

        self.classify_layer = nn.Sequential(
            nn.Linear(int(self.inc1Dim/2), self.num_classes),
            ,nn.BatchNorm1d(self.num_classes)
            #,nn.Sigmoid()
            )

    def forward(self, x):
        # convert x into (batch, self.initial_channels, self.window_size)
        x = x.transpose(1,2)
        xc1 = self.conv_layer1(x)
        #xc1 = self.pool1(xc1)
        xc2 = self.conv_layer2(xc1)
        #xc2 = self.pool2(xc2)
        flattened_x = xc2.view(xc2.size(0), -1)

        xfc1 = self.fc_layer1(flattened_x)
        xfc2 = self.fc_layer2(xfc1)
        prediction = self.classify_layer(xfc2)
        return prediction

class WilkinsAudioCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.initial_channels = 1
        self.conv1_filters = 16
        self.conv2_filters = 8
        self.conv3_filters = 32
        self.max_stride = 8
        self.window_size = 3 * 44100
        self.class_num=10
        self.dropout = config.dropout

        self.layer_seq1 = nn.Sequential(
            nn.Conv1d(in_channels=self.initial_channels,
                out_channels=self.conv1_filters,
                kernel_size=128, stride=1, padding=math.ceil((128-1)/2)),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv1_filters),
            nn.MaxPool1d(kernel_size=64, stride=self.max_stride, padding = math.ceil((64-1)/2))
            )

        self.layer_seq2 = nn.Sequential(
            nn.Conv1d(in_channels=self.conv1_filters,
                out_channels=self.conv2_filters,
                kernel_size=64, stride=1, padding=math.ceil((64-1)/2)),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv2_filters),
            nn.MaxPool1d(kernel_size=64, stride=self.max_stride, padding = math.ceil((64-1)/2))
            )

        self.layer_seq3 = nn.Sequential(
            nn.Conv1d(in_channels=self.conv2_filters,
                out_channels=self.conv3_filters,
                kernel_size=256, stride=1, padding=math.ceil((256-1)/2)),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv3_filters),
            nn.MaxPool1d(kernel_size=64, stride=self.max_stride, padding = math.ceil((64-1)/2))
            )

        self.flat_size = math.ceil(self.window_size / (self.max_stride**3)) * self.conv3_filters

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.flat_size, self.conv3_filters),
            nn.Dropout(self.dropout),
            nn.ReLU()
            )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(self.conv3_filters, self.class_num),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        out_seq1_x = self.layer_seq1(x)
        out_seq2_x = self.layer_seq2(out_seq1_x)
        out_seq3_x = self.layer_seq3(out_seq2_x)
        # retain batch_size dimension, flatten rest
        flattened_x = out_seq3_x.view(out_seq3_x.size(0), -1)
        out_fc1_x = self.fc_layer1(flattened_x)
        out_fc2_x = self.fc_layer2(out_fc1_x)
        return out_fc2_x

class mySpec2cMp2fc(nn.Module):
    def __init__(self, config):
        super(mySpec2cMp2fc, self).__init__()
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.initial_channels=1
        reducedDim=20
        inc1Dim=self.initial_channels*config.dimFactor
        inc2Dim=inc1Dim*config.dimFactor
        classDim=5 # number of classes
        self.window_height = config.n_mels
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.conv1 = nn.Conv2d(self.initial_channels, inc1Dim, kernelSize, padding=paddingSize)
        self.pool1 = nn.MaxPool2d(config.dimFactor, config.dimFactor)
        self.b1 = nn.BatchNorm2d(inc1Dim)
        self.conv2 = nn.Conv2d(inc1Dim, inc2Dim, kernelSize, padding=paddingSize)
        self.pool2 = nn.MaxPool2d(config.dimFactor, config.dimFactor)
        self.b2 = nn.BatchNorm2d(inc2Dim)
        self.fc1 = nn.Linear(inc2Dim * int(self.window_height/(config.dimFactor*2)) * int(self.window_size/(config.dimFactor*2)),reducedDim)
        self.b3 = nn.BatchNorm1d(reducedDim)
        self.fc2 = nn.Linear(reducedDim, classDim)
        self.dimFactor = config.dimFactor
        self.self.initial_channels = self.initial_channels
        self.inc1Dim = inc1Dim
        self.inc2Dim = inc2Dim

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.b1(self.pool1(self.conv1(x))))
        x = F.relu(self.b2(self.pool2(self.conv2(x))))
        x = x.view(-1, self.inc2Dim * int(self.window_height/(self.dimFactor*2)) * int(self.window_size/(self.dimFactor*2)))
        x = F.relu(self.b3(self.fc1(x)))
        x = self.fc2(x)
        return x

class mySpec2cMp1fc(nn.Module):
    def __init__(self, config):
        super(mySpec2c2fc, self).__init__()
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.initial_channels=1
        inc1Dim=self.initial_channels*config.dimFactor
        inc2Dim=inc1Dim*config.dimFactor
        classDim=5 # number of classes
        self.window_height = config.n_mels
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.conv1 = nn.Conv2d(self.initial_channels, inc1Dim, kernelSize, padding=paddingSize)
        self.pool1 = nn.MaxPool2d(config.dimFactor, config.dimFactor)
        self.b1 = nn.BatchNorm2d(inc1Dim)
        self.conv2 = nn.Conv2d(inc1Dim, inc2Dim, kernelSize, padding=paddingSize)
        self.pool2 = nn.MaxPool2d(config.dimFactor, config.dimFactor)
        self.b2 = nn.BatchNorm2d(inc2Dim)
        self.fc1 = nn.Linear(inc2Dim * int(self.window_height/(config.dimFactor*2)) * int(self.window_size/(config.dimFactor*2)), classDim)
        self.dimFactor = config.dimFactor
        self.self.initial_channels = self.initial_channels
        self.inc1Dim = inc1Dim
        self.inc2Dim = inc2Dim

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.b1(self.pool1(self.conv1(x))))
        x = F.relu(self.b2(self.pool2(self.conv2(x))))
        x = x.view(-1, self.inc2Dim * int(self.window_height/(self.dimFactor*2)) * int(self.window_size/(self.dimFactor*2)))
        x = self.fc1(x)
        return x

#class wilkinsTimeDom(nn.Module):
#   def __init__(self, config):
#       super(Wilkins, self).__init__()

        

class LuoNN2cMp2fc(nn.Module):
    def __init__(self, config):
        super(LuoNN2cMp2fc, self).__init__()
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.initial_channels=config.n_mels
        inc1Dim=self.initial_channels*config.dimFactor
        inc2Dim=inc1Dim*config.dimFactor
        classDim=5 # number of classes
        self.window_size = config.window_size
        self.conv1 = nn.Conv1d(self.initial_channels, inc1Dim, kernelSize, padding=paddingSize)
        self.pool1 = nn.MaxPool1d(config.dimFactor, config.dimFactor)
        self.b1 = nn.BatchNorm1d(inc1Dim)
        self.conv2 = nn.Conv1d(inc1Dim, inc2Dim, kernelSize, padding=paddingSize)
        self.pool2 = nn.MaxPool1d(config.dimFactor, config.dimFactor)
        self.b2 = nn.BatchNorm1d(inc2Dim)
        self.fc1 = nn.Linear(inc2Dim * int(config.window_size/(config.dimFactor*2)),inc2Dim)
        self.b3 = nn.BatchNorm1d(inc2Dim)
        self.fc2 = nn.Linear(inc2Dim, classDim)
        self.dimFactor = config.dimFactor
        self.self.initial_channels = self.initial_channels
        self.inc1Dim = inc1Dim
        self.inc2Dim = inc2Dim

    def forward(self, x):
        x = F.relu(self.b1(self.pool1(self.conv1(x))))
        x = F.relu(self.b2(self.pool2(self.conv2(x))))
        x = x.view(-1, self.inc2Dim * int(self.window_size/(self.dimFactor*2)))
        x = F.relu(self.b3(self.fc1(x)))
        x = self.fc2(x)
        return x


class LuoNN2c2fc(nn.Module):
    def __init__(self, config):
        super(LuoNN2c2fc, self).__init__()
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.initial_channels=config.n_mels
        productDim=self.initial_channels*config.dimFactor
        classDim=5 # number of classes
        self.window_size = config.window_size
        self.conv1 = nn.Conv1d(self.initial_channels, productDim, kernelSize, padding=paddingSize)
        self.b1 = nn.BatchNorm1d(productDim)
        self.conv2 = nn.Conv1d(productDim, productDim, kernelSize, padding=paddingSize)
        self.b2 = nn.BatchNorm1d(productDim)
        self.fc1 = nn.Linear(productDim*config.window_size, productDim)
        self.b3 = nn.BatchNorm1d(productDim)
        self.fc2 = nn.Linear(productDim, classDim)
        self.self.initial_channels = self.initial_channels

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.b1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.b2(x))
        x = x.view(-1, self.self.initial_channels*self.window_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.b3(x))
        x = self.fc2(x)
        return x
        
class LuoNN2c2fcNewOrder(nn.Module):
    def __init__(self, config):
        super(LuoNN2c2fcNewOrder, self).__init__()
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.initial_channels=config.n_mels
        productDim=self.initial_channels*config.dimFactor
        reducedDim=5 # number of classes
        self.window_size = config.window_size
        self.conv1 = nn.Conv1d(self.initial_channels, productDim, kernelSize, padding=paddingSize)
        self.b1 = nn.BatchNorm1d(productDim)
        self.conv2 = nn.Conv1d(productDim, productDim, kernelSize, padding=paddingSize)
        self.b2 = nn.BatchNorm1d(productDim)
        self.fc1 = nn.Linear(productDim*config.window_size, productDim)
        self.b3 = nn.BatchNorm1d(productDim)
        self.fc2 = nn.Linear(productDim, reducedDim)
        self.self.initial_channels = self.initial_channels

    def forward(self, x):
        x = F.relu(self.b1(self.conv1(x)))
        x = F.relu(self.b2(self.conv2(x)))
        x = x.view(-1, self.self.initial_channels*self.window_size)
        x = F.relu(self.b3(self.fc1(x)))
        x = self.fc2(x)
        return x


class LuoNN1c2fc(nn.Module):
    def __init__(self, config):
        super(LuoNN1c2fc, self).__init__()
        kernelSize=3
        paddingSize=int((kernelSize-1)/2)
        self.initial_channels=config.n_mels
        productDim=self.initial_channels*config.dimFactor
        reducedDim=5 # number of classes
        self.self.initial_channels = self.initial_channels
        self.window_size = config.window_size
        self.conv1 = nn.Conv1d(self.initial_channels, productDim, kernelSize, padding=paddingSize)
        self.b1 = nn.BatchNorm1d(productDim)
        self.fc1 = nn.Linear(productDim*config.window_size, productDim)
        self.b3 = nn.BatchNorm1d(productDim)
        self.fc2 = nn.Linear(productDim, reducedDim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.b1(x))
        x = x.view(-1, self.self.initial_channels*self.window_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.b3(x))
        x = self.fc2(x)
        return x



def my_loss_function(prediction, target):
    loss = F.cross_entropy(prediction, target)
    return loss
