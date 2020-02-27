
# coding: utf-8

# In[17]:

"""
Summary:  Prepare data. 
Author:   Ningning Pan
Created:  08/05/2019
Modified: - 
"""
import os
import soundfile
import numpy as np
import argparse
import csv
import time
import random
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import _pickle as cPickle
import h5py
import pdb
from sklearn import preprocessing
import import_ipynb
import pdb
import librosa
import multiprocessing as mp
from functools import partial

import prepare_data as pp_data
# import conf as cfg


# In[ ]:

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

###


# In[ ]:

def rms(y):
    """Root mean square. 
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      snr: float, SNR. 
      method: 'rms'. 
      
    Outputs:
      float, scaler. 
    """

    signal_scaling_factor = rms(s) / (rms(n)+0.0001)*10.** (-float(snr)/20.)
    return signal_scaling_factor

    


# In[ ]:

def gen_mix_and_features(workspace, speech_dir, noise_dir, data_type, fs, reuse):
    """generate mixture and get features and targets for DNN training. 
    Then write features and audios to disk.
    How to generate mixtures? We will reuse speech in training set for #reuse times. 
    The time domain speech will firstly multiply a varied random scalar 
    between -26dB to 3dB so that DNN could generalize to volume change in practice.
    Each speech will be added to #reuse noise types 
    which is scalarized by a randomly chosen SNR between -10dB to 15dB.
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'.
      fs: int, sampling rate
      reuse: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when reuse=3, then 4620
          speech with create 4620*3 mixtures. reuse should not larger 
          than the species of noises. 
    """

    target_fs = fs
    
    speech_names = [na for na in os.listdir(speech_dir) if na.lower().endswith(".wav")]
    noise_names = [na for na in os.listdir(noise_dir) if na.lower().endswith(".wav")]
    
    rs = np.random.RandomState(0)
    
    cnt_sp = 0
    t1 = time.time()
    
    for speech_na in speech_names:
        cnt_noi = 1
        # Read speech. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path,target_fs)
        rms_speech = rms(speech_audio) # the root mean square = sqrt(signal power); signal power = mean(signal energy)
        len_speech = len(speech_audio)
        
        # For training data, mix each speech with randomly picked #reuse noises. 
        if data_type == 'train':
            selected_noise_names = rs.choice(noise_names, size=reuse, replace=False) 
            # rs.choice is a function to generate random series, replace=False means that there're no repeats in random series.
        # For test data, mix each speech with all noises. 
        elif data_type == 'test':
            selected_noise_names = noise_names
        else:
            raise Exception("data_type must be train | test!")

        # Mix one speech with different noises many times. 
        for noise_na in selected_noise_names:
            noise_path = os.path.join(noise_dir, noise_na)
            (noise_audio, _) = read_audio(noise_path, target_fs)
            
            len_noise = len(noise_audio)

            if len_noise <= len_speech:
                noise_onset = 0
                noise_offset = len_speech
                n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
                noise_audio_ex = np.tile(noise_audio, n_repeat) #np.title is a repeat function
                noise_audio = noise_audio_ex[noise_onset : noise_offset]
            # If noise is longer than speech then randomly select a segment of noise. 
            else:
                noise_onset = random.randint(0, len_noise - len_speech)
                noise_offset = noise_onset + len_speech
                noise_audio = noise_audio[noise_onset : noise_offset]
                          
            # modify the time domain magnitude of the clean speech signal, which should be in [-26dB,-3dB]
            mag_dB = random.randint(-26,-3) 
            scalar1 = 10**(mag_dB/10)/(rms_speech+0.0001)
            speech_audio1 = scalar1*speech_audio
            
            # randomly choose an SNR in [-10, 15]
            if data_type == 'train':
                iSNR = random.randint(-10, 15) 
            else:
                iSNR_list = [-5, 0, 5, 10, 15]
                iSNR = rs.choice(iSNR_list, size=1, replace=True)
            
            scalar2 = get_amplitude_scaling_factor(speech_audio1, noise_audio, snr=iSNR)
            noise_audio = scalar2*noise_audio
            mixed_audio = speech_audio1 + noise_audio
            
            if np.isnan(np.sum(speech_audio1)) or np.isnan(np.sum(noise_audio)):
                pdb.set_trace()
            ##################################### write audio .... 
            if data_type == 'train':
                out_path = os.path.join(workspace, "Audios", data_type)
            else:
                out_path = os.path.join(workspace, "Audios", data_type, "TIMIT", "%s" % os.path.splitext(noise_na)[0], 
                                        "%ddb" % int(iSNR))
            
            name_save = os.path.join("sp%s_noi%s_%ddb_%d" % 
                (os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0], int(iSNR), int(cnt_noi)))
            out_mix_path = os.path.join(out_path, "mix", "%s.wav" % name_save)
            create_folder(os.path.dirname(out_mix_path))
            write_audio(out_mix_path, mixed_audio, fs)
            
            out_clean_path = os.path.join(out_path, "clean", "%s.wav" % name_save)
            create_folder(os.path.dirname(out_clean_path))
            write_audio(out_clean_path, speech_audio1, fs)
            
            out_noise_path = os.path.join(out_path, "noise", "%s.wav" % name_save)
            create_folder(os.path.dirname(out_noise_path))
            write_audio(out_noise_path, noise_audio, fs)
            
            
            cnt_noi += 1
            
        if cnt_sp % 500 == 0:
            print(cnt_sp)                
        cnt_sp += 1


# In[ ]:

def gen_mix_and_features_multiprocess(speech_na, workspace, speech_dir, noise_dir, data_type, fs, reuse):
    """generate mixture and get features and targets for DNN training. 
    Then write features and audios to disk.
    How to generate mixtures? We will reuse speech in training set for #reuse times. 
    The time domain speech will firstly multiply a varied random scalar 
    between -26dB to 3dB so that DNN could generalize to volume change in practice.
    Each speech will be added to #reuse noise types 
    which is scalarized by a randomly chosen SNR between -10dB to 15dB.
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      fs: int, sampling rate
      reuse: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when reuse=3, then 4620
          speech with create 4620*3 mixtures. reuse should not larger 
          than the species of noises.

    Notice: multiprocess will take all cpus.
    """
    target_fs = fs
    
    noise_names = [na for na in os.listdir(noise_dir) if na.lower().endswith(".wav")]
    cnt_noi = 1
    # Read speech. 
    speech_path = os.path.join(speech_dir, speech_na)
    (speech_audio, _) = read_audio(speech_path,target_fs)
    rms_speech = rms(speech_audio) # the root mean square = sqrt(signal power); signal power = mean(signal energy)
    len_speech = len(speech_audio)
    
    np.random.seed()
    # For training data, mix each speech with randomly picked #reuse noises. 
    if data_type == 'train':
        selected_noise_names = np.random.choice(noise_names, size=reuse, replace=False) 
        print(selected_noise_names[0])
        # rs.choice is a function to generate random series, replace=False means that there're no repeats in random series.
        # For test data, mix each speech with all noises. 
    elif data_type == 'test':
        selected_noise_names = noise_names
    else:
        raise Exception("data_type must be train | test!")

        # Mix one speech with different noises many times. 
    for noise_na in selected_noise_names:
        noise_path = os.path.join(noise_dir, noise_na)
        (noise_audio, _) = read_audio(noise_path, target_fs)
            
        len_noise = len(noise_audio)

        if len_noise <= len_speech:
            noise_onset = 0
            noise_offset = len_speech
            n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
            noise_audio_ex = np.tile(noise_audio, n_repeat) #np.title is a repeat function
            noise_audio = noise_audio_ex[noise_onset : noise_offset]
            # If noise is longer than speech then randomly select a segment of noise. 
        else:
            noise_onset = np.random.randint(0, len_noise - len_speech, size=1)[0]
            noise_offset = noise_onset + len_speech
            noise_audio = noise_audio[noise_onset : noise_offset]
                          
            # modify the time domain magnitude of the clean speech signal, which should be in [-26dB,-3dB]
        mag_dB = np.random.randint(-22,-3) 
        scalar1 = 10**(mag_dB/10)/(rms_speech+0.0001)
        speech_audio1 = scalar1*speech_audio
            
            # randomly choose an SNR in [-10, 15]
        if data_type == 'train':
            iSNR = np.random.randint(-10, 15) 
        else:
            iSNR_list = [-5, 0, 5, 10, 15]
            iSNR = np.random.choice(iSNR_list, size=1, replace=True)
#             print(iSNR)
            
        scalar2 = get_amplitude_scaling_factor(speech_audio1, noise_audio, snr=iSNR)
        noise_audio = noise_audio*scalar2
        mixed_audio = speech_audio1 + noise_audio
            
        if np.isnan(np.sum(speech_audio1)) or np.isnan(np.sum(noise_audio)):
            continue
            
        ##################################### write audio .... 
        if data_type == 'train':
            out_path = os.path.join(workspace, "Audios", data_type)
        else:
            out_path = os.path.join(workspace, "Audios", data_type, "TIMIT", "%s" % os.path.splitext(noise_na)[0], 
                                    "%ddb" % int(iSNR))

        name_save = os.path.join("sp%s_noi%s_%ddb_%d" % 
            (os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0], int(iSNR), int(cnt_noi)))
        out_mix_path = os.path.join(out_path, "mix", "%s.wav" % name_save)
        create_folder(os.path.dirname(out_mix_path))
        write_audio(out_mix_path, mixed_audio, fs)

        out_clean_path = os.path.join(out_path, "clean", "%s.wav" % name_save)
        create_folder(os.path.dirname(out_clean_path))
        write_audio(out_clean_path, speech_audio1, fs)

        out_noise_path = os.path.join(out_path, "noise", "%s.wav" % name_save)
        create_folder(os.path.dirname(out_noise_path))
        write_audio(out_noise_path, noise_audio, fs)


        cnt_noi += 1


# In[ ]:

if __name__ == '__main__':
    
    # STEP 1: Create mixtures
    workspace = '../workspace' # root directory for training dataset
    speech_dir_train = '../../dataset/TIMIT/TRAIN/' #original clean speech dataset for training
    noise_dir_train = '../../dataset/NOISE/freesound.org/' #original noise dataset for training
    speech_dir_test = '../dataset/MRT_mTurk/' #clean speech for test
    noise_dir_test = '../dataset/NOISE/noiseX92/' # noise dataset for test
    
    target_fs=16000
    reuse = 49 # it is how many times you would use the speech training data repeatedly
    
    ############### training data create ######################
    ##### no multiprocessing
#     gen_mix_and_features(workspace,speech_dir_train,noise_dir_train,'train', fs, reuse)

    ##### multiprocessing to create training data
    speech_names_train = [na for na in os.listdir(speech_dir_train) if na.lower().endswith(".wav")]
    pool = mp.Pool(mp.cpu_count())
  
    partial_gen_mix_and_features_train = partial(gen_mix_and_features_multiprocess, workspace = workspace, speech_dir=speech_dir_train,
                                           noise_dir = noise_dir_train, data_type='train', fs=target_fs, reuse=49)
    pool.map(partial_gen_mix_and_features_train, speech_names_train)
    print("train gen mix and features finish")
    
    ################## test data create ####################
    gen_mix_and_features(workspace,speech_dir_test,noise_dir_test, 'test', 'MRT_mTurk',target_fs, 1)
    print("test gen mix and features finish")

