from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
import os, pdb, pickle, random, math, torch

from multiprocessing import Process, Manager


class pathSpecDataset(Dataset):
    """Dataset class for using a path to spec folders,
	path for labels,
	generates random windowed subspec examples,
	associated labels,
	optional conditioning."""
    def __init__(self, config, spmel_params):
        """Initialize and preprocess the dataset."""
        self.spmel_dir = config.data_dir
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second)
        
        style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
        singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
        #self.one_hot_array = np.eye(len(class_names))[np.arange(len(class_names))]
        dir_name, _, fileList = next(os.walk(self.spmel_dir))
        fileList = sorted(fileList)
        dataset = []
        for file_name in fileList:
            if file_name.endswith('.npy'):
                spmel = np.load(os.path.join(dir_name, file_name))
                for style_idx, style_name in enumerate(style_names):
                    if style_name in file_name:
                        for singer_idx, singer_name in enumerate(singer_names):
                            if singer_name in file_name:
                                dataset.append((spmel, style_idx, singer_idx))
                                break
                        break

        self.dataset = dataset
        self.num_specs = len(dataset)

    """__getitem__ selects a speaker and chooses a random subset of data (in this case
    an utterance) and randomly crops that data. It also selects the corresponding speaker
    embedding and loads that up. It will now also get corresponding pitch contour for such a file"""
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.dataset
        # spkr_data is literally a list of skpr_id, emb, and utterances from a single speaker
        spmel, style_idx, singer_idx  = dataset[index]
        # pick random spmel_chunk with random crop


        """old incorrect way of feeding data to network"""
############################################################################
#
#        if spmel.shape[0] < self.window_size:
#            len_pad = self.window_size - spmel.shape[0]
#            spmel_chunk = np.pad(spmel, ((0,len_pad),(0,0)), 'constant')
#            #pitch = np.pad(pitch_info, ((0,len_pad),(0,0)), 'constant')
#        elif spmel.shape[0] > self.window_size:
#            left = np.random.randint(spmel.shape[0]-self.window_size)
#            spmel_chunk = spmel[left:left+self.window_size, :]
#            #pitch = pitch_info[left:left+self.window_size, :]
#        else:
#            spmel_chunk = spmel
#            #pitch = pitch_info
# 
#        return spmel_chunk, style_idx, singer_idx
############################################################################
        
        chunk_num = math.floor(spmel / self.window_size)
        spmel_mini_batches = []
        for i in range(chunk_num)
            offset = i * self.window_size
#            if offset > (spmel.shape[0] - self.window_size):
#                len_pad = (offset + self.window_size) - spmel.shape[0]
#                batch = spmel[offset:]
#                batch = np.pad(batch, ((0,lenPad),(0,0)), 'constant')
#            else:
            batch = spmel[offset : offset+self.window_size]
            spmel_mini_batches.append(batch)

        return spmel_mini_batches, style_idx, singer_idx

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
        #self.one_hot_array = np.eye(len(class_names))[np.arange(len(class_names))]
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
