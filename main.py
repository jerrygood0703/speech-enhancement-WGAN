from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import sys

from data_utils import *
from ops import *
from model import *
from dnn_model import *
from cwgan import *

use_waveform = True
batch_size = 64
learning_rate = 1e-4
iters = 45000

mode = sys.argv[1] # stage1, stage2, test
log_path = 'stage2_log/'
model_path = 'stage1_model/model_20170802/'
model_path2 = 'stage2_model/model_20170802/'
test_path = model_path2 # switch between stage1 and stage2
test_list = "/mnt/gv0/user_sylar/segan_data/noisy_test_list"
record_name = "/data_wave.tfrecord"

if use_waveform:
    G=Generator((1,2,3),(1,2,3))
    D=Discriminator()
else:
    G=spec_Generator((1,2,3),(1,2,3))
    D=spec_Discriminator()

def check_dir(path_name):
    if tf.gfile.Exists(path_name):
        print('Folder already exists: {}\n'.format(path_name))
    else:
        tf.gfile.MkDir(path_name)

check_dir(model_path)
check_dir(model_path2)

with tf.device('cpu'):
    reader = dataPreprocessor(record_name, use_waveform=use_waveform)
    clean, noisy = reader.read_and_decode(batch_size=batch_size,num_threads=32)
#with tf.device('gpu'):
gan = GradientPenaltyWGAN(G,D,noisy,clean,log_path,model_path,model_path2,use_waveform,lr=learning_rate)

if mode=='test':
    if use_waveform:
        x_test = tf.placeholder("float", [None, 1, 16384, 1], name='test_noisy')
    else:
        x_test = tf.placeholder("float", [None, 1, 257, 32], name='test_noisy')
    gan.test(x_test, test_path, test_list)
else:
    gan.train(mode, iters)
