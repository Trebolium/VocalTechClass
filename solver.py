import models, utils
import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random, pdb

class VoiceTechniqueClassifier:
    def __init__(self, config, spmel_params):
        """ initialise configurations"""
        self.config = config
        self.device = torch.device(f'cuda:{self.config.which_cuda}' if torch.cuda.is_available() else 'cpu')

        if config.is_wilkins == True:
            self.model = models.WilkinsAudioCNN(config)
        else:
            self.model = models.Luo2019AsIs(config, spmel_params)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.reg)
        if self.config.load_ckpt != '':
            g_checkpoint = torch.load(self.config.load_ckpt)
            self.model.load_state_dict(g_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(g_checkpoint['self.optimizer_state_dict'])
            # pdb.set_trace()
            # fixes tensors on different devices error
            # https://github.com/pytorch/pytorch/issues/2830
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.config.which_cuda)
            self.previous_ckpt_iters = g_checkpoint['epoch']
        else:
            self.previous_ckpt_iters = 0 
        self.model.to(self.device)

    def infer(self, epoch, loader, history_list, writer, examples_per_epoch, mode):

        def batch_iterate():

            print(f'=====> {split_name}: ')
            accum_loss = 0
            accum_corrects = 0
            for batch_num, (x_data, y_data, singer_id)  in enumerate(loader):

                x_data = x_data.to(self.device, dtype=torch.float)
                y_data = y_data.to(self.device)

                prediction = self.model(x_data)
                loss = nn.functional.cross_entropy(prediction, y_data) 
                _, predicted = torch.max(prediction.data, 1)
                corrects = (predicted == y_data).sum().item()
                accum_corrects += corrects
                accuracy = corrects / y_data.shape[0]
                accum_loss += loss.item()

                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                    # inaccurate reading here if on last batch and drop_last=False
                    epoch,
                    batch_num * self.config.batch_size,
                    examples_per_epoch,
                    100. * batch_num / len(loader),
                    loss.item(),
                    accuracy)) # calculates average loss per example

                y_data = np.expand_dims(y_data.cpu(),1)
                singer_id = np.expand_dims(singer_id.cpu(),1)
                if batch_num == 0:
                    labels = np.hstack((y_data,singer_id))
                else:
                    tmp =  np.hstack((y_data,singer_id)) 
                    labels = np.vstack((labels, tmp))

            return labels, accum_loss, accum_corrects
        
        if mode == 'train':
            self.model.train()
            loss_hist=history_list[0]
            acc_hist=history_list[1]
            split_name = 'train'
            labels, accum_loss, accum_corrects = batch_iterate()
        elif mode == 'eval':
            self.model.eval()
            loss_hist=history_list[2]
            acc_hist=history_list[3]
            split_name = 'test'
            with torch.no_grad():
                labels, accum_loss, accum_corrects = batch_iterate()

        epoch_loss = accum_loss / len(loader)
        epoch_accuracy = accum_corrects / examples_per_epoch
        if self.config.is_wilkins:
            writer.add_scalar(f"Number Correct/{split_name}", accum_corrects, epoch)
            writer.add_scalar(f"Accuracy/{split_name}", epoch_accuracy, epoch)
            writer.add_scalar(f"Loss/{split_name}", epoch_loss, epoch)
            writer.add_histogram(f"layer_seq1.bias", self.model.layer_seq1[0].bias, epoch)
            writer.add_histogram(f"layer_seq1.weight", self.model.layer_seq1[0].weight, epoch)
            writer.add_histogram(f"layer_seq1.weight.grad", self.model.layer_seq1[0].weight.grad, epoch) 
        else:
            writer.add_scalar(f"Number Correct/{split_name}", accum_corrects, epoch)
            writer.add_scalar(f"Accuracy/{split_name}", epoch_accuracy, epoch)
            writer.add_scalar(f"Loss/{split_name}", epoch_loss, epoch)
#            writer.add_histogram(f"enc_convs_conv_layer1", self.model.enc_convs[0][0].bias, epoch)
#            writer.add_histogram(f"enc_convs_conv_layer1.weight", self.model.enc_convs[0][0].weight, epoch)
#            writer.add_histogram(f"enc_convs_conv_layer1.weight.grad", self.model.enc_convs[0][0].weight.grad, epoch) 
        print()
        print('Epoch {} Loss: {:.4f}, Acc: {:.4f}'.format(epoch, epoch_loss, epoch_accuracy))
        loss_hist.append(epoch_loss)
        acc_hist.append(epoch_accuracy)
        save_path = './results/' +self.config.file_name +'/' +str(epoch) +'Epoch_checkpoint.pth.tar'
        self.save_checkpoints(epoch, epoch_loss, epoch_accuracy, save_path)

        return labels

    def save_checkpoints(self, epoch, loss, accuracy, save_path):
        if epoch == self.config.epochs or epoch % self.config.ckpt_freq == 0:
            print('saving model')
            checkpoint = {'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'accuracy': accuracy}
            torch.save(checkpoint, save_path)

