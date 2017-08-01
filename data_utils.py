import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal
import scipy.io.wavfile as wav
import librosa
import random
from sklearn import preprocessing
import tensorflow as tf

def wav2spectrum(wavfile, data_shape):
    y,sr=librosa.load(wavfile,sr=16000)
    training_data = np.empty((1000, 257, 7)) # For Noisy data
    D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    Sxx=np.log10(abs(D)**2)
    Sxx=preprocessing.scale(Sxx,axis=1)
    idx = 0     
    for i in range(3, Sxx.shape[1]-3): # 7 Frmae
        training_data[idx,:,:] = Sxx[:,i-3:i+4] # For Noisy data
        idx = idx + 1
    return training_data[:idx]

def data_generator(noisy_path, clean_path, batch_size, data_shape):
    noisy_list = []
    clean_list = []
    with open(noisy_path,'r') as f:
        for line in f:
            noisy_list.append(line[:-1])
    with open(clean_path,'r') as f:
        for line in f:
            clean_list.append(line[:-1])
    assert(len(noisy_list) == len(clean_list))

    list_total = len(noisy_list)
    list_idx = 0
    count = 0
    idx = 0

    noisy_data = wav2spectrum(noisy_list[list_idx], data_shape)
    clean_data = wav2spectrum(clean_list[list_idx], data_shape)
    list_idx += 1
    while True:
        if list_idx == list_total:
            list_idx = 0

        source = np.zeros((batch_size, 1, noisy_data.shape[1], noisy_data.shape[2]))
        target = np.zeros((batch_size, 1, clean_data.shape[1], clean_data.shape[2]))
        while count < batch_size:
            if idx == noisy_data.shape[0]:
                noisy_data = wav2spectrum(noisy_list[list_idx], data_shape)
                clean_data = wav2spectrum(clean_list[list_idx], data_shape)
                list_idx += 1
                idx = 0

            source[count,0] = noisy_data[idx]
            target[count,0] = clean_data[idx]
            count+=1
            idx+=1

        count = 0
        yield source, target


FRAMELENGTH = 16384
OVERLAP = FRAMELENGTH//2

class WavGenerator(object):
    def __init__(self, data_list, use_waveform):
        self.wav_list = []
        self.use_waveform = use_waveform
        self.idx = 0
        self.total_data = 0
        with open(data_list,'r') as f:
            for line in f:
                self.wav_list.append(line[:-1])
        random.shuffle(self.wav_list) 
        self.total_data = len(self.wav_list)

        print('wav [{}]'.format(self.total_data)) 

    def __call__(self, bs):
        count = 0
        wav=[]
        while count<bs:
            y,sr=librosa.load(self.wav_list[self.idx],sr=16384)
            # zero padding
            temp = np.zeros(((y.shape[0]//FRAMELENGTH+1)*FRAMELENGTH))
            temp[:y.shape[0]] = y
            y = temp

            self.idx+=1
            if self.idx == self.total_data: self.idx = 0
            if self.use_waveform==True:
                for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
                    wav.append(y[i:i+FRAMELENGTH])
                    count+=1
                    if count==bs:break

        return np.array(wav).reshape((-1,1,FRAMELENGTH,1))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def slice_signal(signal, window_size, overlap):
#     """ Return windows of the given signal by sweeping in stride fractions
#         of window
#     """
#     n_samples = signal.shape[0]
#     offset = overlap
#     slices = []
#     for beg_i, end_i in zip(range(0, n_samples, offset),
#                             range(window_size, n_samples + offset,
#                                   offset)):
#         if end_i - beg_i < window_size:
#             break
#         slice_ = signal[beg_i:end_i]
#         if slice_.shape[0] == window_size:
#             slices.append(slice_)
#     return np.array(slices, dtype=np.int32)

# def read_and_slice(filename, FRAMELENGTH, OVERLAP):
#     sr, wav_data = wav.read(filename)
#     if sr != 16000:
#         raise ValueError('Sampling rate is expected to be 16kHz!')
#     temp = np.zeros(((wav_data.shape[0]//FRAMELENGTH+1)*FRAMELENGTH))
#     temp[:wav_data.shape[0]] = wav_data
#     wav_data = temp
#     signals = slice_signal(wav_data, FRAMELENGTH, OVERLAP)
#     return signals  

class dataPreprocessor(object):
    def __init__(self, record_name, noisy_filelist=None, clean_filelist=None, use_waveform=True):
        self.noisy_filelist = noisy_filelist
        self.clean_filelist = clean_filelist
        self.use_waveform = use_waveform
        self.record_path = "/mnt/gv0/user_sylar/segan_data"
        self.record_name = record_name

        if use_waveform:
            self.FRAMELENGTH = 16384
            self.OVERLAP = self.FRAMELENGTH//2
        else:
            self.FRAMELENGTH = 32
            self.OVERLAP = self.FRAMELENGTH//2

        if noisy_filelist and clean_filelist:
            print('Write records')
        else:
            print('Read-only mode')

    def slice_signal(self, signal, window_size, overlap):
        """ Return windows of the given signal by sweeping in stride fractions
            of window
        """
        n_samples = signal.shape[0]
        offset = overlap
        slices = []
        for beg_i, end_i in zip(range(0, n_samples, offset),
                                range(window_size, n_samples + offset,
                                      offset)):
            if end_i - beg_i < window_size:
                break
            slice_ = signal[beg_i:end_i]
            if slice_.shape[0] == window_size:
                slices.append(slice_)
        return np.array(slices, dtype=np.int32)

    def read_and_slice(self, filename):
        sr, wav_data = wav.read(filename)
        print(wav_data)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        signals = self.slice_signal(wav_data, self.FRAMELENGTH, self.OVERLAP)
        return signals  

    def make_spectrum(self, filename, use_normalize):
        sr, y = wav.read(filename)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype!='float32':
            y = np.float32(y/32767.)

        D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) 
        if use_normalize:
            mean = np.mean(Sxx, axis=1).reshape((257,1))
            std = np.std(Sxx, axis=1).reshape((257,1))+1e-12
            Sxx = (Sxx-mean)/std  
        slices = []
        for i in range(0, Sxx.shape[1]-self.FRAMELENGTH, self.OVERLAP):
            slices.append(Sxx[:,i:i+self.FRAMELENGTH])
        return np.array(slices)   

    def write_tfrecord(self):
        if self.noisy_filelist is None or self.clean_filelist is None:
            raise ValueError('Read-only mode\nCreate an instance by specifying a filename.')

        if tf.gfile.Exists(self.record_path):
            print('Folder already exists: {}\n'.format(self.record_path))
        else:
            tf.gfile.MkDir(self.record_path)

        nlist = [x[:-1] for x in open(self.noisy_filelist).readlines()]
        clist = [x[:-1] for x in open(self.clean_filelist).readlines()]

        assert len(nlist) == len(clist)

        out_file = tf.python_io.TFRecordWriter(self.record_path+self.record_name)

        if self.use_waveform:
            for n,c in zip(nlist, clist):
                print(n,c)
                wav_signals = self.read_and_slice(c)
                noisy_signals = self.read_and_slice(n)
                print(wav_signals)            
                for (wav, noisy) in zip(wav_signals, noisy_signals):
                    wav_raw = wav.tostring()
                    noisy_raw = noisy.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'wav_raw': _bytes_feature(wav_raw),
                        'noisy_raw': _bytes_feature(noisy_raw)}))
                    out_file.write(example.SerializeToString())
            out_file.close()

        else:
            for n,c in zip(nlist, clist):
                print(n,c)
                wav_signals = self.make_spectrum(c, False)
                noisy_signals = self.make_spectrum(n, True)
                for (wav, noisy) in zip(wav_signals, noisy_signals):
                    print(wav.shape, noisy.shape)
                    wav_raw = wav.tostring()
                    noisy_raw = noisy.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'wav_raw': _bytes_feature(wav_raw),
                        'noisy_raw': _bytes_feature(noisy_raw)}))
                    out_file.write(example.SerializeToString())
            out_file.close()             

    def read_and_decode(self,batch_size=16, num_threads=8):
        filename_queue = tf.train.string_input_producer([self.record_path+self.record_name])        
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'wav_raw': tf.FixedLenFeature([], tf.string),
                    'noisy_raw': tf.FixedLenFeature([], tf.string),
                })
        if self.use_waveform:
            wave = tf.decode_raw(features['wav_raw'], tf.int32)
            wave.set_shape(self.FRAMELENGTH)
            wave = (2./65535.) * tf.cast((wave - 32767), tf.float32) + 1.
            wave = tf.reshape(wave, [1,-1,1])
            noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
            noisy.set_shape(self.FRAMELENGTH)
            noisy = (2./65535.) * tf.cast((noisy - 32767), tf.float32) + 1.
            noisy = tf.reshape(noisy, [1,-1,1])
        else:
            wave = tf.decode_raw(features['wav_raw'], tf.float32)
            wave = tf.reshape(wave,[1,257,self.FRAMELENGTH])
            noisy = tf.decode_raw(features['noisy_raw'], tf.float32)
            noisy = tf.reshape(noisy,[1,257,self.FRAMELENGTH])

        wavebatch, noisybatch = tf.train.shuffle_batch([wave,
                                             noisy],
                                             batch_size=batch_size,
                                             num_threads=num_threads,
                                             capacity=1000 + 3 * batch_size,
                                             min_after_dequeue=1000,
                                             name='wav_and_noisy')
        return wavebatch, noisybatch
 