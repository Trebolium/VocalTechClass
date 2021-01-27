import csv, pickle, os, pdb
import numpy as np
import matplotlib.pyplot as plt

def saveHistory(history_list, dir_path, string_config, epoch_labels):

    upper_lim = 0.5
    #pdb.set_trace()
    train_loss_hist = np.asarray(history_list[0])
    val_loss_hist = np.asarray(history_list[2])
    train_acc_hist = np.asarray(history_list[1])
    val_acc_hist = np.asarray(history_list[3])
    epochs = np.arange(len(train_acc_hist))

    plt.figure()
    plt.subplot(422)
    plt.title(string_config)
    plt.ylim(0,upper_lim)
    #plt.yticks(np.arange(0, 1, step=0.2))
    plt.plot(epochs,train_loss_hist,'r--',label='train loss')
    plt.plot(epochs,val_loss_hist,'b--',label='val loss')
    plt.legend()

    plt.subplot(424)
    plt.xlabel("Epochs")
    plt.ylim(0,upper_lim)
    #plt.yticks(np.arange(0, 1, step=0.2))
    plt.plot(epochs,train_acc_hist,'c--',label='train acc')
    plt.plot(epochs,val_acc_hist,'g--',label='val acc')
    plt.legend()

    plt.subplot(425)
    plt.ylim(0,upper_lim)
    #plt.yticks(np.arange(0, 1, step=0.2))
    plt.plot(epochs,train_acc_hist,'c--',label='train acc')
    plt.legend()

    plt.subplot(427)
    plt.xlabel("Epochs")
    plt.ylim(0,upper_lim)
    #plt.yticks(np.arange(0, 1, step=0.2))
    plt.plot(epochs,val_acc_hist,'g--',label='val acc')
    plt.legend()

    plt.subplot(421)
    plt.ylim(0,0.1)
    #plt.yticks(np.arange(0, 1, step=0.2))
    plt.plot(epochs,train_loss_hist,'r--',label='train loss')
    plt.legend()

    plt.subplot(423)
    plt.ylim(0,0.1)
    #plt.yticks(np.arange(0, 1, step=0.2))
    plt.plot(epochs,val_loss_hist,'b--',label='val loss')
    plt.legend()

    plt.show()

    plt.savefig(dir_path +'/history_plot.png')

    with open(dir_path +'/history_log.csv', "w") as csvFile:
        writer = csv.writer(csvFile)
        header = 'TrainLoss','ValLoss','TrainAcc','ValAcc'
        writer.writerow(header)
        for epoch in range(len(train_acc_hist)):
            writer.writerow((train_loss_hist[epoch], val_loss_hist[epoch], train_acc_hist[epoch], val_acc_hist[epoch]))
        # writer.writerow(('Time Taken:', str(seconds), ' seconds'))
        min_val_index = np.where(val_loss_hist==np.amin(val_loss_hist))
        writer.writerow(('Lowest val loss/epoch:', min(val_loss_hist), int(min_val_index[0])))
    csvFile.close()
    
    with open(dir_path +'/label_log.pkl', 'wb') as handle:
        pickle.dump(epoch_labels, handle)
