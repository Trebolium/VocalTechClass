import csv, pickle, random, os, json, torch, pdb
import numpy as np
import matplotlib.pyplot as plt


# returns a list of filepaths collected from a parent directory and all subdirectories
def recursive_file_retrieval(parent_path):
    file_path_list = []
    dir_list = []
    parent_paths = [parent_path]
    more_subdirs = True
    while more_subdirs == True:
        subdir_paths = [] 
        for i, parent_path in enumerate(parent_paths):
            dir_list.append(parent_path)
            r,dirs,files = next(os.walk(parent_path)) 
            for f in files:
                file_path_list.append(os.path.join(r,f))
            # if there are more subdirectories
            if len(dirs) != 0:
                for d in dirs:
                    subdir_paths.append(os.path.join(r,d))
                # if we've finished going through subdirectories (each parent_path), stop that loop
            if i == len(parent_paths)-1:
                # if loop about to finish, change parent_paths content and restart loop
                if len(subdir_paths) != 0:
                    parent_paths = subdir_paths
                else:
                    more_subdirs = False
    return dir_list, file_path_list


# convert string input into bool
def str2bool(v):
    return v.lower() in ('true')


# generate random splits for singers
def get_vocalset_splits(this_seed):
    m_list = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_']
    f_list = ['f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
    random.seed(this_seed)
    random.shuffle(m_list)
    random.shuffle(f_list)
    train_m_list, test_m_list = (m_list[:-3],m_list[-3:])
    train_f_list, test_f_list = (f_list[:-2],f_list[-2:])
    train_list = train_m_list + train_f_list
    test_list = test_m_list + test_f_list
    return train_list, test_list


# generate random hyperparameters for searching best configurations
def random_params(config):
    config.lr = random.uniform(1e-6, 1e-2)
    config.n_mels = int(random.uniform(10, 79))
    config.dropout = random.uniform(0, 0.8)
    config.batch_size = random.randit(0, 4)
    config.reg = random.uniform(0, 0.01),
    config.chunk_seconds = int(random.uniform(23, 82))
    return config


# save history of predictions and metrics as graphs
def saveHistory(history_list, dir_path, string_config, tech_singer_labels, pred_target_labels):

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
    
    with open(dir_path +'/tech_singer_labels_log.pkl', 'wb') as handle:
        pickle.dump(tech_singer_labels, handle)
    with open(dir_path +'/pred_target_labels_log.pkl', 'wb') as handle:
        pickle.dump(pred_target_labels, handle)


# prepare directories for files affiliated with this model instance
def assert_dirs(config, results_dir):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    file_name_dir = os.path.join(results_dir, config.file_name)
    if not os.path.exists(file_name_dir):
        os.mkdir(file_name_dir)
    return file_name_dir


# analyze configuration parameters to change if necessary, save and return config
def setup_config(config, file_name_dir):
    if config.iteration>1:
        config = random_params(config)
    if config.load_ckpt != '':
        previous_epochs = int(re.findall('\d+', config.load_ckpt)[0])
    else:
        previous_epochs = 0
    # convert config object to string
    string_config = json.dumps(vars(config))[1:-1]
    string_config = ''.join(e for e in string_config if e.isalnum())
    with open(file_name_dir +'/config_params.pkl','wb') as File:
        pickle.dump(config, File) 
    return string_config, previous_epochs, config
    