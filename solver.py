from re import I
import torch, yaml, pdb
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import determine_model, update_model
from utils import determine_writer

class VoiceTechniqueClassifier:
    def __init__(self, config):

        self.config = config
        self.final_epoch = config.epochs+1
        self.device = torch.device(f'cuda:{self.config.which_cuda}' if torch.cuda.is_available() else 'cpu')
        self.save_path = './results/' +self.config.file_name +'/best_epoch_checkpoint.pth.tar'
        model = determine_model(config)
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg)
        self.previous_ckpt_iters, self.model, self.optimizer = update_model(config, model, self.optimizer)
        self.writer = determine_writer(config.file_name)
        self.model.to(self.device)


    # save model checkpoint data
    def save_checkpoints(self, epoch, loss, accuracy):
        print('saving model')
        checkpoint = {'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy}
        torch.save(checkpoint, self.save_path)
        print('model saved!')


    # run forward and backward passes for train and test epochs
    def infer(self, epoch, loader, history_list, examples_per_epoch, mode):

        def batch_iterate():

            # initiate params for this epoch
            print(f'=====> {mode}: ')
            accum_loss = 0
            accum_corrects = 0
            for batch_num, (x_data, y_data, singer_id)  in enumerate(loader):
                
                x_data = x_data.to(self.device, dtype=torch.float)
                y_data = y_data.to(self.device)

                if self.config.model == 'luo' or self.config.model == 'choi_k2c2':
                    #tensors must be reshaped so that they have 3 dims
                    x_data = x_data.view(x_data.shape[0] * x_data.shape[1], x_data.shape[2], x_data.shape[3])
                prediction = self.model(x_data)

                # calculate loss
                loss = nn.functional.cross_entropy(prediction, y_data) 
                _, predicted = torch.max(prediction.data, 1)
                corrects = (predicted == y_data).sum().item()
                accum_corrects += corrects
                accuracy = corrects / y_data.shape[0]
                accum_loss += loss.item()

                # backprop
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Print metric within epoch
                print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}\tCorrect: {:.6f}'.format(
                    # inaccurate reading here if on last batch and drop_last=False
                    epoch,
                    batch_num * self.config.batch_size,
                    examples_per_epoch,
                    100. * batch_num / len(loader),
                    loss.item(),
                    accuracy,
                    corrects)) # calculates average loss per example
                
                # formulate data for saving to history_lists
                y_data = np.expand_dims(y_data.cpu(),1)
                singer_id = np.expand_dims(singer_id.cpu(),1)
                if batch_num == 0:
                    tech_singer_labels = np.hstack((y_data,singer_id))
                    pred_target_labels = np.hstack((predicted.unsqueeze(1).cpu().detach().numpy(), y_data))
                else:
                    tmp =  np.hstack((y_data,singer_id)) 
                    tech_singer_labels = np.vstack((tech_singer_labels, tmp))
                    tmp = np.hstack((predicted.unsqueeze(1).cpu().detach().numpy(), y_data))
                    pred_target_labels = np.vstack((pred_target_labels, tmp))

            return pred_target_labels, tech_singer_labels, accum_loss, accum_corrects
        
        
        # main parent block for training and test epochs
        if mode == 'train':
            self.model.train()
            loss_hist=history_list[0]
            acc_hist=history_list[1]
            pred_target_labels, tech_singer_labels, accum_loss, accum_corrects = batch_iterate()
        elif mode == 'test':
            best_acc = 0
            self.model.eval()
            loss_hist=history_list[2]
            acc_hist=history_list[3]
            with torch.no_grad():
                pred_target_labels, tech_singer_labels, accum_loss, accum_corrects = batch_iterate()

        # calculate averaged epoch metrics print, save to tensorboard, print, update metric history
        epoch_loss = accum_loss / examples_per_epoch
        epoch_accuracy = accum_corrects / examples_per_epoch
        self.writer.add_scalar(f"Accuracy/{mode}", epoch_accuracy, epoch)
        self.writer.add_scalar(f"Loss/{mode}", epoch_loss, epoch)
        print('\nEpoch {} Loss: {:.4f}, Acc: {:.4f} Corrects:{:.4f}'.format(epoch, epoch_loss, epoch_accuracy, accum_corrects))
        loss_hist.append(epoch_loss)
        acc_hist.append(epoch_accuracy)

        # save model if best accuracy
        if mode == 'test':
            self.writer.flush()
            if epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                self.save_checkpoints(epoch, epoch_loss, epoch_accuracy)

        if epoch >= self.final_epoch:
            self.writer.close()

        return pred_target_labels, tech_singer_labels



