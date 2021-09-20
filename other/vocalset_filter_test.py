import os, pdb, argparse, time, shutil, pickle, random 
import numpy as np 
import soundfile as sf 
from scipy import signal 
from scipy.signal import get_window, medfilt 
from numpy.random import RandomState 
 
start_time = time.time() 
 
parser = argparse.ArgumentParser(description='args from main') 
parser.add_argument('--class_dir', type=str, default='singer', help='choose dataset folder from Vocalset to analyze, organised by class name') 
# default values taken from Luo2019 
parser.add_argument('--trg_sr', type=int, default=22050) 
parser.add_argument('--fft_size', type=int, default=1024) 
parser.add_argument('--hop_size', type=int, default=256) 
parser.add_argument('--fmin', type=int, default=16000) 
parser.add_argument('--fmax', type=int, default=16000) 
parser.add_argument('--n_mels', type=int, default=96) 
args = parser.parse_args()

# audio file directory
rootDir = '/import/c4dm-datasets/VocalSet1-2/data_by_' +args.class_dir

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)
file_counter = 0
singer_counter = 0
class_num_per_singer = 10
beltCount = trillCount = straightCount = fryCount = vibratoCount = breathyCount = 0
class_counter_list = [beltCount, trillCount, straightCount, fryCount, vibratoCount, breathyCount]
class_list=['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
exclude_list = ['caro','row','long','dona']

counter_list = []
for subdir_idx, subdir in enumerate(sorted(subdirList)):
    #subdirs could be singers
    print(subdir)
    if 'male1' == subdir:
        class_num_per_singer = 7
    else:
        class_num_per_singer = 10
    singer_counter += 1
    beltCount = trillCount = straightCount = fryCount = vibratoCount = breathyCount = 0
    class_counter_list = [beltCount, trillCount, straightCount, fryCount, vibratoCount, breathyCount]
    subDirName ,subsubdirList, _ = next(os.walk(os.path.join(dirName,subdir)))
    for subsubdir_idx, subsubdir in enumerate(sorted(subsubdirList)):
        #subsubdir could be exercise_type 
        subsubDirName, subsubsubdirList,_ = next(os.walk(os.path.join(subDirName, subsubdir)))
        for subsubsubdir_idx, subsubsubdir in enumerate(sorted(subsubsubdirList)):
            #subsubsubdir could be technique
            _,_, fileList = next(os.walk(os.path.join(subsubDirName,subsubsubdir)))
            for file_idx, fileName in enumerate(sorted(fileList)):
                exclusion_found = False
                # ensure that only mic1 files are processed
                for class_idx, class_name in enumerate(class_list):
                    #pdb.set_trace()
                    if class_name in fileName:
                        for exclusion in exclude_list:
                            if exclusion in fileName:
                                exclusion_found = True
                                break
                        if exclusion_found == True:
                            break #out of class_name loop, as we don't need to consider this file anymore
                        if fileName[-6] != '_':
                            break
                        # if number of classes in singer exceed specified allowance
                        if class_counter_list[class_idx]+1 > class_num_per_singer:
                            break 
                        class_counter_list[class_idx] += 1
                        # do the operation intended for every valid entry
                        print(fileName)
                        file_counter += 1
    counter_list.append((subdir, class_counter_list))
total_per_class = [0,0,0,0,0,0]

for _, class_counter_list in counter_list:
    print(class_counter_list)
#    for cps_idx, class_per_singer in enumerate(class_counter_list):
#        total_per_class[cps_idx] += cps_idx
print('total_per_class', total_per_class)
print('total', file_counter)
print('singer', singer_counter)
#pdb.set_trace()
