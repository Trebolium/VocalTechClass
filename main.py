from utils import saveHistory, str2bool, assert_dirs, setup_config, recursive_file_retrieval, substring_inclusion, substring_exclusion
from solver import VoiceTechniqueClassifier
from data import determine_dataset, get_vocalset_splits
import pickle, argparse, os, csv, pdb
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # string inputs
    parser.add_argument('--config_file', type=str, default='', metavar='N', help='Provide path to config file that could be used to superseed user-specified params')
    parser.add_argument('--file_name', type=str, default='defaultName', metavar='N', help='Make a name for the model your training')
    parser.add_argument('--model', type=str, default='choi_k2c2', help='adjust code to work with Wilkins model and audio not spectrograms')
    parser.add_argument('--load_ckpt', type=str, default='', metavar='N', help='path to previously trained network weights')
    parser.add_argument('--data_dir', type=str, default='./example_ds', metavar='N', help='path to dataset')
    parser.add_argument('--test_list', type=str, default='', metavar='N', help='if there is a preferred split to use, specify here')
    # integer inputs
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 4)')
    parser.add_argument('--lstm_num', type=int, default=2, metavar='N', help='2 if not specified')
    parser.add_argument('--chunk_num', type=int, default=6, metavar='N', help='chunk_seconds is 23 if not specified')
    parser.add_argument('--which_cuda', type=int, default=0, metavar='N', help='determine which GPU to use')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--ckpt_freq', type=int, default=100, metavar='N', help='number of times to make a safety save of the model')
    parser.add_argument('--seed', type=int, default=1, metavar='N', help='randomisation seed')
    parser.add_argument('--iteration', type=int, default=1, metavar='N', help='make this more than one if using random search for ideal h_params')
    parser.add_argument('--n_mels', type=int, default=96, metavar='N', help='freq dim for features being used')
    # float inputs
    parser.add_argument('--chunk_seconds', type=float, default=0.5, metavar='N', help='duration that input features are truncated to')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='N', help='learning rate')
    parser.add_argument('--reg', type=float, default=0, metavar='N', help='determines amount of weight decay for regularisation')
    parser.add_argument('--dropout', type=float, default=0., metavar='N', help='amount of dropout')
    # boolean inputs
    parser.add_argument('--use_attention', type=str2bool, default=False, help='toggle use of attention mechanism')
    parser.add_argument('--is_blstm', type=str2bool, default=True, help='boolean for whether lstm or blstm used')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    config = parser.parse_args()

    # determine whether to use user-specified or previous config params
    if config.config_file != '':
        config = pickle.load(open(config.config_file, 'rb'))

    # set variables
    torch.manual_seed(1)
    config.cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{config.which_cuda}" if config.cuda else "cpu")
    results_dir = 'results'
    file_name_dir = assert_dirs(config, results_dir)
    results_csv = os.path.join(results_dir, config.file_name, 'RandomSearchReport.csv')

    # save all model history to csv file
    with open(results_csv, "w") as csvResults:
        csv_writer = csv.writer(csvResults)
        header = 'Title','lr','melNum','dropout','batchSize','reg','windowSize','bestEpoch','Loss','Acc'
        csv_writer.writerow(header)
        
        # for each interation, a new set of hyperparameters will be generated (if i>1)
        for i in range(config.iteration):

            train_list, test_list = get_vocalset_splits(1)
            config.test_list = ' '.join(test_list)
            string_config, previous_epochs, config = setup_config(config, file_name_dir) #create new h_params in this function

            #generate file list
            style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
            singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
            _, fileList = recursive_file_retrieval(config.data_dir)
            fileList = [f for f in fileList if f.endswith('.wav') or f.endswith('.npy')]
            fileList = substring_inclusion(fileList, style_names)
            fileList = substring_exclusion(fileList, ['long', 'slow'])
            fileList.sort()
            
            dataset = determine_dataset(config, fileList, style_names, singer_names)

            
            # split file list into test/train indices list for Dataloaders
            train_indices_list = [fileName_idx for fileName_idx, fileName in enumerate(fileList) for substring in train_list if substring in fileName]
            test_indices_list = [fileName_idx for fileName_idx, fileName in enumerate(fileList) for substring in test_list if substring in fileName]

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
                train_pred_target_labels, train_tech_singer_labels = vt_classer.infer(epoch, train_loader, history_list, len(train_indices_list), 'train')
                val_pred_target_labels, val_tech_singer_labels = vt_classer.infer(epoch, test_loader, history_list, len(test_indices_list), 'test')
                tech_singer_labels.append((train_tech_singer_labels, val_tech_singer_labels)) 
                pred_target_labels.append((train_pred_target_labels, val_pred_target_labels)) 

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


