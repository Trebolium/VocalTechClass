from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
import os, pdb, pickle, random, math, torch, sys
from utils import recursive_file_retrieval, substring_inclusion

"""Gets singer_id, style_labels and feature_arrays from file_names"""
class pathSpecDataset(Dataset):

    def __init__(self, config, spmel_params):

        # setup attributes and 
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second)
        self.chunk_num = config.chunk_num
        style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
        singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
        _, fileList = recursive_file_retrieval(config.data_dir)
        fileList = sorted([file for file in fileList if file.endswith('npy')])
        fileList = substring_inclusion(fileList, style_names)
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
        # pick random spmel_chunk with random crop

        """Ensure all spmels are the length of (self.window_size * chunk_num)"""
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



def get_loader(config, index_for_splits, num_workers=0):
    """Build and return a data loader."""

    dataset = Utterances(config, index_for_splits)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader

class audioSnippetDataset(Dataset):
    def __init__(self, config):
        self.audio_paths_file = config.data_dir
        self.window_size = 3* 44100
        style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
        singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
        dataset = []
        filtered_audio_path_list = pickle.load(open(self.audio_paths_file, 'rb')) 
        sorted_filtered_audio_path_list = sorted(filtered_audio_path_list)
        sorted_filtered_audio_file_list = [os.path.basename(x) for x in sorted_filtered_audio_path_list]
        for file_idx, file_name in enumerate(sorted_filtered_audio_file_list):
            for style_idx, style_name in enumerate(style_names):
                if style_name in file_name:
                    for singer_idx, singer_name in enumerate(singer_names):
                        if singer_name in file_name:
                            audio, sr = sf.read(sorted_filtered_audio_path_list[file_idx])
                            dataset.append((audio, style_idx, singer_idx))
                            break
                    break

        self.dataset = dataset
        self.num_audios = len(dataset)

    def __getitem__(self, index):
        dataset = self.dataset
        audio, style_idx, singer_idx = dataset[index]
        #pdb.set_trace()
        if audio.shape[0] < self.window_size:
            len_pad = self.window_size - audio.shape[0]
            audio_chunk = np.pad(audio, ((0,len_pad),(0,0)), 'constant')
        elif audio.shape[0] > self.window_size:
            left = np.random.randint(audio.shape[0]-self.window_size)
            audio_chunk = audio[left:left+self.window_size]
        else:
            audio_chunk = audio
        return audio_chunk, style_idx, singer_idx

    def __len__(self):
        return self.num_audios
