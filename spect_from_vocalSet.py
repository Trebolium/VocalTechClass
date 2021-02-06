import yaml, os, pdb, argparse, time, shutil, librosa, pickle, random
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
from scipy import signal
from scipy.signal import get_window, medfilt
from librosa.filters import mel
from numpy.random import RandomState
#from yin import pitch_calc

start_time = time.time()

parser = argparse.ArgumentParser(description='args from main')
parser.add_argument('--class_dir', type=str, default='singer', help='choose dataset folder from Vocalset to analyze, organised by class name')
parser.add_argument('--trg_data_dir', type=str, default='./deletable')
# default values taken from Luo2019
parser.add_argument('--trg_sr', type=int, default=22050)
parser.add_argument('--fft_size', type=int, default=1024)
parser.add_argument('--hop_size', type=int, default=256)
parser.add_argument('--fmin', type=int, default=90)
parser.add_argument('--n_mels', type=int, default=96)
args = parser.parse_args()

# The Butterworth filter is a type of signal processing filter designed to have as flat frequency
# response as possible (no ripples) in the pass-band and zero roll off response in the stop-band. 
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_size=args.fft_size, hop_size=args.hop_size):
    x = np.pad(x, int(fft_size//2), mode='reflect')
    noverlap = fft_size - hop_size
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_size, fft_size)
    strides = x.strides[:-1]+(hop_size*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    fft_window = get_window('hann', fft_size, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_size).T    
    return np.abs(result)    

def remove_quiet_edges(src_path):
    # From https://stackoverflow.com/questions/29547218/
    audio_seg = AudioSegment.from_file(src_path, format='wav')
    audio_seg = effects.normalize(audio_seg)
    duration = len(audio_seg)
    start_trim = detect_leading_silence(audio_seg)
    end_trim = detect_leading_silence(audio_seg.reverse())
    trimmed_audio_seg = audio_seg[start_trim:duration-end_trim]
    trimmed_audio_seg.export('./silence_trimmed_audio.wav', format="wav")

def detect_leading_silence(audio_seg, silence_thresh=-60.0, chunk_size=10):
    # From https://stackoverflow.com/questions/29547218/
    silence_found = False
    while silence_found == False:
        silence_thresh -=2
        for time_step in range(0,len(audio_seg),chunk_size):
            if audio_seg[time_step:time_step+chunk_size].dBFS > silence_thresh:
                silence_found = True
                break
    return time_step
        


def preprocess(audio_sr_tuple): 
    audio, sr = audio_sr_tuple
    # Remove drifting noise
    y = signal.filtfilt(b, a, audio)
    # Ddd a little random noise for model roubstness
    wav = y * 0.96 + (RandomState(1).rand(y.shape[0])-0.5)*1e-06
    # resample 48kHz to 16kHz
    resampled_wav = librosa.resample(wav, sr, args.trg_sr)
    # compute pitch contour
    #timestamp, frequency_prediction, confidence, activation = crepe.predict(resampled_wav, 16000, viterbi=False, step_size=16)
    # preprocess pitch contour
    #one_hot_preprocessed_pitch_contours, voiced_log_freq, uttr_var, uttr_std, uttr_mean = pitch_preprocessing(frequency_prediction, confidence) 
    # Compute spect
    D = pySTFT(resampled_wav).T
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    #Author mentioned min level -100 and ref level 16 dB in https://github.com/auspicious3000/autovc/issues/4
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)    
    return S

# audio file directory
rootDir = '/import/c4dm-datasets/VocalSet1-2/data_by_' +args.class_dir
# spectrogram directory
# pitch contour directory
# targetDirPitch = './pitch'

print('Deleting old directories...')
#for directory in [args.trg_data_dir, targetDirPitch]:
if os.path.exists(args.trg_data_dir):
    shutil.rmtree(args.trg_data_dir)
os.makedirs(args.trg_data_dir)

dict_file = dict(n_mels = args.n_mels,
            fft_size = args.fft_size,
            sr = args.trg_sr,
            hop_size = args.hop_size,
            fmin = args.fmin)

with open(args.trg_data_dir +'/spmel_params.yaml', 'w') as File:
    documents = yaml.dump(dict_file, File)

#pdb.set_trace()
_, src_sr = sf.read('/import/c4dm-datasets/VocalSet1-2/data_by_singer/male1/arpeggios/belt/m1_arpeggios_belt_a.wav')
mel_basis = mel(args.trg_sr, args.fft_size, fmin=args.fmin, fmax=args.trg_sr/2, n_mels=args.n_mels).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, args.trg_sr, order=5)

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)
file_counter = 0
singer_counter = 0
class_num_per_singer = 10
beltCount = trillCount = straightCount = fryCount = vibratoCount = breathyCount = 0
class_counter_list = [beltCount, trillCount, straightCount, fryCount, vibratoCount, breathyCount]
class_list=['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
exclude_list = ['caro','row','long','dona']

for subdir_idx, subdir in enumerate(sorted(subdirList)):
    #print(subdir)
    #os.makedirs(os.path.join(args.trg_data_dir,subdir))
    #os.makedirs(os.path.join(targetDirPitch,subdir))
    # adjustments for imbalanced male1 data
    if 'male1' == subdir:
        class_num_per_singer = 7
    else:
        class_num_per_singer = 10
    singer_counter += 1
    beltCount = trillCount = straightCount = fryCount = vibratoCount = breathyCount = 0
    class_counter_list = [beltCount, trillCount, straightCount, fryCount, vibratoCount, breathyCount]
    
    """Navigate through VocalSet directory network"""
    subDirName ,subsubdirList, _ = next(os.walk(os.path.join(dirName,subdir)))
    for subsubdir_idx, subsubdir in enumerate(sorted(subsubdirList)):
        #subsubdir could be exercise_type 
        #print(subsubdir)
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

                        """Now that directories have been navigated and files have been filtered, do processing """
                        # Read audio file
                        path_name = os.path.join(dirName, subdir, subsubdir, subsubsubdir, fileName)
                        print('converting: ', path_name)
                        #if fileName == 'f1_arpeggios_vocal_fry_e.wav':
                        #    pdb.set_trace()
                        remove_quiet_edges(path_name)
                        #pdb.set_trace()
                        preprocessed_data = preprocess(sf.read('./silence_trimmed_audio.wav'))
                        # save spect    
                        np.save(os.path.join(args.trg_data_dir, fileName[:-4]),
                                preprocessed_data.astype(np.float32), allow_pickle=False)    
                        # save pitch contour
                        #with open(os.path.join(targetDirPitch, subdir, fileName[:-5]) +'.pkl','wb') as handle:
                         #       pickle.dump([one_hot_preprocessed_pitch_contours, voiced_log_freq, uttr_var, uttr_std, uttr_mean], handle)
                        #np.save(os.path.join(targetDirPitch, subdir, fileName[:-5]),
                         #       one_hot_preprocessed_pitch_contours.astype(np.float32), allow_pickle=False)
            #            if counter==2:
            #                break
                        file_counter += 1

#with open('./spkr_id_var_std_mean.pkl', 'wb') as handle:
    #pickle.dump(spkr_mean_stats, handle)


print('time taken', time.time()-start_time)
