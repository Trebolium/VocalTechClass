from utils import saveHistory, get_vocalset_splits, str2bool, assert_dirs, random_params, setup_config, recursive_file_retrieval
from solver import VoiceTechniqueClassifier
from data import pathSpecDataset, audioSnippetDataset
import pickle, argparse, yaml, os, csv, sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_os import recursive_file_retrieval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='', metavar='N')
    parser.add_argument('--file_name', type=str, default='defaultName', metavar='N')
    parser.add_argument('--model', type=str, default='choi_k2c2', help='adjust code to work with Wilkins model and audio not spectrograms')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 4)')
    parser.add_argument('--lstm_num', type=int, default=2, metavar='N', help='2 if not specified')
    parser.add_argument('--use_attention', type=str2bool, default=False, help='')
    parser.add_argument('--is_blstm', type=str2bool, default=True, help='')
    parser.add_argument('--chunk_num', type=int, default=6, metavar='N', help='chunk_seconds is 23 if not specified')
    parser.add_argument('--which_cuda', type=int, default=0, metavar='N')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='N')
    parser.add_argument('--seed', type=int, default=1, metavar='N')
    parser.add_argument('--dropout', type=float, default=0., metavar='N')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--chunk_seconds', type=float, default=0.5, metavar='N', help='chunk_seconds is 0.5 if not specified')
    parser.add_argument('--ckpt_freq', type=int, default=100, metavar='N')
    parser.add_argument('--load_ckpt', type=str, default='', metavar='N')
    parser.add_argument('--reg', type=float, default=0, metavar='N')
    parser.add_argument('--iteration', type=int, default=1, metavar='N')
    parser.add_argument('--n_mels', type=int, default=96, metavar='N')
    parser.add_argument('--data_dir', type=str, default='./example_ds', metavar='N')
    parser.add_argument('--test_list', type=str, default='', metavar='N')
    config = parser.parse_args()

    # determine whether to use current or previous config params
    if config.config_file != '':
        config = pickle.load(open(config.config_file, 'rb'))

    # set variables
    torch.manual_seed(1)
    config.cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{config.which_cuda}" if config.cuda else "cpu")
    results_dir = 'results'
    file_name_dir = assert_dirs(config, results_dir)
    
    # prepare csv file and 
    results_csv = os.path.join(results_dir, config.file_name, 'RandomSearchReport.csv')
    with open(results_csv, "w") as csvResults:
        csv_writer = csv.writer(csvResults)
        header = 'Title','lr','melNum','dropout','batchSize','reg','windowSize','bestEpoch','Loss','Acc'
        csv_writer.writerow(header)
        
        # for each interation, a new set of hyperparameters will be generated (if i>1)
        for i in range(config.iteration):

            train_list, test_list = get_vocalset_splits(1)
            config.test_list = ' '.join(test_list)
            string_config, previous_epochs, config = setup_config(config, file_name_dir) #create new h_params in this function

            # generate dataset and file_paths
            if config.model == 'wilkins': # get list from pickle object if using Wilkins' proposed network
                config.data_dir = './audio_path_list.pkl'
                dataset = audioSnippetDataset(config)
                fileList = pickle.load(open(config.data_dir, 'rb'))
            else: # if not Wilkins' network, assume to be using mel-spectrogram features
                with open(config.data_dir +'/feat_params.yaml') as File:
                    spmel_params = yaml.load(File, Loader=yaml.FullLoader)
                _, fileList = recursive_file_retrieval(config.data_dir)
                dataset = pathSpecDataset(config, spmel_params)
            file_path_list = sorted(fileList)
            fileList = [os.path.basename(x) for x in file_path_list if x.endswith('.npy')]
            
            # split file list into test/train indices list for Dataloaders
            train_indices_list = [fileName_idx for fileName_idx, fileName in enumerate(fileList) for substring in train_list if substring in fileName]
            test_indices_list = [fileName_idx for fileName_idx, fileName in enumerate(fileList) for substring in test_list if substring in fileName]

            if config.file_name == 'defaultName' or config.file_name == 'deletable':
                writer = SummaryWriter('testRuns/test')
            else:
                writer = SummaryWriter(comment = '_' +config.file_name)

            tech_singer_labels = []
            pred_target_labels = []
            train_sampler = SubsetRandomSampler(train_indices_list) 
            test_sampler = SubsetRandomSampler(test_indices_list)   
            train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=False)
            test_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=test_sampler, shuffle=False, drop_last=False)
            vt_classer = VoiceTechniqueClassifier(config)

            # history_list records predictions, targets, accuracy, loss, and stored for later analysis
            history_list=[[], [], [], []]
            # iterate through train and test set cyclically
            for epoch in range(previous_epochs+1, previous_epochs+config.epochs+1):
                train_pred_target_labels, train_tech_singer_labels = vt_classer.infer(epoch, train_loader, history_list, writer, len(train_indices_list), 'train')
                val_pred_target_labels, val_tech_singer_labels = vt_classer.infer(epoch, test_loader, history_list, writer, len(test_indices_list), 'test')
                tech_singer_labels.append((train_tech_singer_labels, val_tech_singer_labels)) 
                pred_target_labels.append((train_pred_target_labels, val_pred_target_labels)) 
                writer.flush()

            writer.close() 


            saveHistory(history_list, file_name_dir, string_config, tech_singer_labels, pred_target_labels)
            bestLoss = history_list[0][0]

            for idx, loss in enumerate(history_list[0]):
                if loss <= bestLoss:
                    bestLoss=loss
                    bestEpoch=idx
            
            bestAcc = history_list[3][bestEpoch]
            
            csv_writer.writerow((string_config, config.lr, config.n_mels, config.dropout, config.batch_size, config.reg, config.chunk_seconds, bestEpoch, bestLoss, bestAcc))
    csvResults.close(
    )


