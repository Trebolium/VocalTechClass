from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
import random, math, yaml, pdb
from utils import recursive_file_retrieval, substring_inclusion


"""Gets singer_id, style_labels and feature_arrays from file_names"""
class pathSpecDataset(Dataset):

    def __init__(self, config, fileList, style_names, singer_names):

        # setup attributes and
        with open(config.data_dir +'/feat_params.yaml') as File:
            spmel_params = yaml.load(File, Loader=yaml.FullLoader)
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second)
        self.chunk_num = config.chunk_num

        dataset = []

        # iterate through files and save with appropriate labels
        for file_path in fileList:
            spmel = np.load(file_path)
            for style_idx, style_name in enumerate(style_names):
                if style_name in file_path:
                    for singer_idx, singer_name in enumerate(singer_names):
                        if singer_name in file_path:
                            dataset.append((spmel, style_idx, singer_idx))
                            break
                    break
        
        self.dataset = dataset
        self.num_specs = len(dataset)

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.dataset
        # spkr_data is literally a list of skpr_id, emb, and utterances from a single speaker
        spmel, style_idx, singer_idx  = dataset[index]
        #Ensure all spmels are the length of (self.window_size * chunk_num)
        chunk_num = self.chunk_num
        desired_spmel_length = (self.window_size * chunk_num)
        difference = spmel.shape[0] - desired_spmel_length
        offset = random.randint(0, difference)
        length_adjusted_spmel = spmel[offset : offset + desired_spmel_length]

        # stack spmel as even chunks
        for i in range(chunk_num):
            offset = i * self.window_size
            if i == 0:
                cat_batch = length_adjusted_spmel[offset : offset+self.window_size]
                cat_batch = np.expand_dims(cat_batch, 0)
            else:
                tmp = length_adjusted_spmel[offset : offset+self.window_size]
                tmp = np.expand_dims(tmp, 0)
                cat_batch = np.concatenate((cat_batch, tmp), 0)
                
        return cat_batch, style_idx, singer_idx

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_specs



"""used when wanting to use audio data only"""
class audioSnippetDataset(Dataset):

    def __init__(self, fileList, style_names, singer_names):

        self.window_size = 3 * 44100
        # check file_paths for matching singer and style strings, save when found
        dataset = []
        for file_idx, file_name in enumerate(fileList):
            for style_idx, style_name in enumerate(style_names):
                if style_name in file_name:
                    for singer_idx, singer_name in enumerate(singer_names):
                        if singer_name in file_name:
                            audio, _ = sf.read(fileList[file_idx])
                            dataset.append((audio, style_idx, singer_idx))
                            break
                    break
        self.dataset = dataset
        self.num_audios = len(dataset)

    def __getitem__(self, index):
        dataset = self.dataset
        audio, style_idx, singer_idx = dataset[index]
        
        # randomly truncate audio
        if audio.shape[0] < self.window_size:
            len_pad = self.window_size - audio.shape[0]
            audio_chunk = np.pad(audio, ((0,len_pad),(0,0)), 'constant')
        elif audio.shape[0] >= self.window_size:
            left = np.random.randint(audio.shape[0]-self.window_size)
            audio_chunk = audio[left:left+self.window_size]

        return audio_chunk, style_idx, singer_idx

    def __len__(self):
        return self.num_audios


# choose dataset based on desired model formant
def determine_dataset(config, fileList, style_names, singer_names):
    if config.model == 'wilkins':
        dataset = audioSnippetDataset(fileList, style_names, singer_names)
    else:
        dataset = pathSpecDataset(config, fileList, style_names, singer_names)
    return dataset


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