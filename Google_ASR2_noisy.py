
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:51:55 2015

@author: Jason
"""
import num2words
import speech_recognition as sr
import os
import numpy as np
from wer import wer

r = sr.Recognizer()

# enhanced_dir = "../../segan/data/enhanced_test"
enhanced_dir = "./stage2_model/model_20170720/20170720_90kep"
clean_dir = "/mnt/gv0/user_sylar/segan_data/clean_test_wav_16k"

file_name = os.listdir(clean_dir)

wer_list = []

for f in file_name:
    print(clean_dir+'/'+f, enhanced_dir+'/'+f)
    with sr.WavFile(clean_dir+'/'+f) as source:
        clean_audio = r.record(source)
    with sr.WavFile(enhanced_dir+'/'+f) as source2:
        enhanced_audio = r.record(source2)    
    while True:
        try:
            gt=r.recognize_google(clean_audio)
            result=r.recognize_google(enhanced_audio)
            encoded_gt = gt.encode('utf8').split()
            encoded_r = result.encode('utf8').split()
            print(encoded_gt,encoded_r)
            score = wer(encoded_gt,encoded_r)
            wer_list.append(score)
            print(score)
            break
        except sr.RequestError:                                  # No internet connection
            print("No internet connection")
        except sr.UnknownValueError:
            print("Sorry sir, but, I could not understand what you said!")
            wer_list.append(100.0)
            break

print(np.mean(np.array(wer_list)))