from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *
import re
import os

class GradientPenaltyWGAN(object):
    '''
    Wasserstein GAN with Gradient Penalty (conditional version)
    
    # ==== DISCUSSION ====
    # J: Can we force D(x) to be around zero to avoid the "constant shift" problem?
    #    https://github.com/igul222/improved_wgan_training/issues/17
    # A: Yes, but this doesn't seem to be important. (?)
    l_mean = -1. * tf.reduce_mean(
        GaussianLogDensity(
            x=c_real,
            mu=tf.zeros_like(c_real),
            log_var=tf.zeros_like(c_real))
    ) 
    loss['l_D'] = - loss['W_dist'] + lam * gp + l_mean

    # Another unattempted trial was: 
    # penalty = tf.square(tf.nn.relu(grad_norm - 1.)) # FIXME: experimental
    '''
    def __init__(self, g_net, d_net, 
                data_mask, data_noisy, data_clean, 
                log_path, model_path, model_path2,
                use_waveform, 
                lr=1e-4, gan_lamb=1.):

        self.model_path = model_path
        self.model_path2 = model_path2
        self.log_path = log_path

        self.lamb_gp = 10.
        self.lamb_recon = 100.
        self.gan_lamb = gan_lamb
        self.lr = lr
        self.g_net = g_net
        self.d_net = d_net
        self.noisy = data_noisy # noisy data shape [batch_size,1,257,8*8]
        self.clean = data_clean		     # # clean data shape [batch_size,1,257,8*8]
        self.mask = data_mask
        self.enhanced = self.g_net(self.noisy, reuse=False)

        self.d_real = self.d_net(self.noisy, self.mask, reuse=False)
        self.d_fake = self.d_net(self.noisy, self.enhanced, reuse=True)
        e = tf.random_uniform([tf.shape(self.noisy)[0], 1, 1, 1], 0., 1., name='epsilon')
        x_intp = self.mask + e * (self.enhanced - self.mask) 
        d_intp = self.d_net(self.noisy, x_intp, reuse=True)
        self.gp = self._compute_gradient_penalty(d_intp, x_intp)

        self.loss = dict()
        self.loss['E_real'] = tf.reduce_mean(self.d_real)
        self.loss['E_fake'] = tf.reduce_mean(self.d_fake)
        self.loss['G_recon'] = tf.reduce_mean(tf.squared_difference(self.enhanced, self.mask))
        # self.loss['G_mask'] = tf.reduce_mean(tf.squared_difference(self.masked, self.clean))
        self.loss['W_dist'] = self.loss['E_real'] - self.loss['E_fake']
        self.loss['l_G'] = self.gan_lamb * (-self.loss['E_fake']) + self.lamb_recon * (self.loss['G_recon'] )#+ self.loss['G_mask'])
        self.loss['l_D'] = self.gan_lamb * (-self.loss['W_dist'] + self.lamb_gp * self.gp)

        # # For summaries
        # with tf.name_scope('Summary'):
        E_real = tf.summary.scalar('E_real', self.loss['E_real'])
        E_fake = tf.summary.scalar('E_fake', self.loss['E_fake'])
        l_D = tf.summary.scalar('l_D', self.loss['l_D'])
        G_recon = tf.summary.scalar('G_recon', self.loss['G_recon'])
        # G_mask = tf.summary.scalar('G_mask', self.loss['G_mask'])
        l_gp = tf.summary.scalar('gp', self.gp)
        W_dist = tf.summary.scalar('W_dist', self.loss['W_dist'])
        #tf.summary.histogram('d_real', self.d_real)
        #tf.summary.histogram('d_fake', self.d_fake)
        if use_waveform:
            audio_summ = tf.summary.audio('enhanced', tf.reshape(self.enhanced,(int(self.enhanced.get_shape()[0]), -1)), 16000)
            self.g_summs = [E_fake, G_recon, audio_summ]
        else:
            self.g_summs = [E_fake, G_recon]#, G_mask]
        self.d_summs = [E_real, E_fake, l_D, l_gp, W_dist]

        self.d_opt, self.g_opt = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9)\
                .minimize(self.loss['l_D'], var_list=self.d_net.vars)
            self.g_opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9)\
                .minimize(self.loss['l_G'], var_list=self.g_net.vars)
            # self.d_opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)\
            #     .minimize(self.loss['l_D'], var_list=self.d_net.vars)
            # self.g_opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)\
            #     .minimize(self.loss['l_G'], var_list=self.g_net.vars)
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement = True)
        config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=config)

    def _compute_gradient_penalty(self, J, x, scope_name='GradientPenalty'):
        ''' Gradient Penalty
        Input:
            `J`: the loss
            `x`: shape = [b, c, h, w]
        '''
        with tf.name_scope(scope_name):
            grad = tf.gradients(J, x)[0]  # as the output is a list, [0] is needed
            grad_square = tf.square(grad)
            grad_squared_norm = tf.reduce_sum(grad_square, axis=[1, 2, 3])
            grad_norm = tf.sqrt(grad_squared_norm)
            # penalty = tf.square(tf.nn.relu(grad_norm - 1.)) # FIXME: experimental
            penalty = tf.square(grad_norm - 1.)
        return tf.reduce_mean(penalty)

    def train(self, mode="stage1", iters=65000):

        self.sess.run(tf.global_variables_initializer())

        if tf.gfile.Exists(self.log_path+"D"):
            tf.gfile.DeleteRecursively(self.log_path+"D")
        tf.gfile.MkDir(self.log_path+"D")

        if tf.gfile.Exists(self.log_path+"G"):
            tf.gfile.DeleteRecursively(self.log_path+"G")
        tf.gfile.MkDir(self.log_path+"G")

        g_merged = tf.summary.merge(self.g_summs)
        d_merged = tf.summary.merge(self.d_summs)
        D_writer = tf.summary.FileWriter(self.log_path+"D", self.sess.graph)
        G_writer = tf.summary.FileWriter(self.log_path+"G", self.sess.graph)

        if mode == 'stage1':
            save_path = self.model_path
            print('Training:stage1')

        elif mode == 'stage2':
            save_path = self.model_path2
            print('Training:stage2')

            with open(self.model_path + "checkpoint", 'r') as f:
                line = f.readline()
            latest_step = re.sub("[^0-9]", "", line)
            print(latest_step)
            with tf.device("/cpu:0"):
                saver = tf.train.Saver()
                saver.restore(self.sess, self.model_path + "model-" + latest_step)
        #-----------------------------------------------------------------#
        saver = tf.train.Saver(max_to_keep=10)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)
        try:
            while not coord.should_stop():
                for i in range(iters):
                    if i%100==0:
                        _, summary, loss_d = self.sess.run([self.d_opt, d_merged, self.loss['l_D']])    
                        D_writer.add_summary(summary, i)
                        _, summary, loss_g = self.sess.run([self.g_opt, g_merged, self.loss['l_G']])
                        G_writer.add_summary(summary, i)
                        print("\rIter:{} LD:{} LG:{}".format(i, loss_d, loss_g))
                    else:
                        for _ in range(5):
                            _ = self.sess.run([self.d_opt])  
                        _ = self.sess.run([self.g_opt])

                    if i % 5000 == 4999:
                        saver.save(self.sess, save_path + 'model', global_step=i)
                    if i == iters-1:
                        coord.request_stop()

        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit')
        finally:
            coord.request_stop()
        coord.join(threads)

        return
        
    def test(self, x_test, test_path, test_list):

        print('Testing:'+test_path)
        import scipy.io.wavfile as wav
        import librosa
        import scipy
        def slice_signal(signal, window_size, overlap):
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

        def make_spectrum_phase(y, FRAMELENGTH, OVERLAP):
            D = librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
            Sxx = np.log10(abs(D)**2) 
            phase = np.exp(1j * np.angle(D))
            mean = np.mean(Sxx, axis=1).reshape((257,1))
            std = np.std(Sxx, axis=1).reshape((257,1))+1e-12
            Sxx = (Sxx-mean)/std  
            return Sxx, phase, mean, std

        def recons_spec_phase(Sxx_r, phase):
            Sxx_r = np.sqrt(10**Sxx_r)
            R = np.multiply(Sxx_r , phase)
            result = librosa.istft(R,
                             hop_length=256,
                             win_length=512,
                             window=scipy.signal.hamming)
            return result

        if x_test.get_shape()[3] > 1:
            FRAMELENGTH = 64
            OVERLAP = 2
        else:
            FRAMELENGTH = 16384
            OVERLAP = 16384  

        # Load model          
        with open(test_path + "checkpoint", 'r') as f:
            line = f.readline()
        latest_step = re.sub("[^0-9]", "", line)
        print(latest_step)
        with tf.device("/cpu:0"):
            saver = tf.train.Saver()
            saver.restore(self.sess, test_path + "model-" + latest_step)

        enhanced = self.g_net(x_test, reuse=True)

        nlist = [_[:-1] for _ in open(test_list).readlines()]
        for name in nlist:
            sr, y = wav.read(name)
            if sr != 16000:
                raise ValueError('Sampling rate is expected to be 16kHz!')
            # For spectrum data
            if x_test.get_shape()[3] > 1:
                if y.dtype!='float32':
                    y = np.float32(y/32767.)
                spec, phase, mean, std = make_spectrum_phase(y, FRAMELENGTH, OVERLAP)
                print(spec.shape)

                # Padding spectrum
                pad_num = OVERLAP * np.ceil((spec.shape[1] - FRAMELENGTH)*1. / OVERLAP) + FRAMELENGTH
                temp = np.zeros((257, int(pad_num)))
                temp[:, :spec.shape[1]] = temp[:, :spec.shape[1]] + spec
                # print(temp.shape)

                slices = []
                for i in range(0, spec.shape[1]-FRAMELENGTH, OVERLAP):
                    slices.append(spec[:,i:i+FRAMELENGTH])
                slices = np.array(slices).reshape((-1,1,257,FRAMELENGTH))
                output = self.sess.run(enhanced,{x_test:slices})
                # print(output.shape)


                sample_weight = np.zeros((257, spec.shape[1]))
                temp_out = np.zeros((257, spec.shape[1]))
                count = 0
                for i in range(0, temp.shape[1]-FRAMELENGTH, OVERLAP):
                    temp_out[:,i:i+FRAMELENGTH] += (np.log10(output[count,0,:,:]) + temp[:,i:i+FRAMELENGTH])
                    sample_weight[:,i:i+FRAMELENGTH] += 1
                    count+=1
                temp_out = temp_out / sample_weight
                print(temp_out.shape)
                # print(i)
                # The un-enhanced part of spec should be un-normalized
                # spec[:, i:] = (spec[:, i:] * std) + mean

                recons_y = recons_spec_phase(temp_out, phase)             
                y_out = librosa.util.fix_length(recons_y, y.shape[0])
                temp_name = name.split('/')
                wav.write(os.path.join(test_path,"enhanced",temp_name[-3],temp_name[-2],temp_name[-1]),16000,np.int16(y_out*32767))
                # wav.write(test_path+"enhanced/"+name.split('/')[-1],16000,np.int16(y_out*32767))
            # For waveform data
            else:
                # temp = np.zeros(((y.shape[0]//FRAMELENGTH+1)*FRAMELENGTH))
                # temp[:y.shape[0]] = y
                # wav_data = temp
                # signals = slice_signal(wav_data, FRAMELENGTH, OVERLAP)
                # wave = (2./65535.) * (signals.astype(np.float32)-32767) + 1.
                # print(np.max(wave),np.min(wave))
                # wave = np.reshape(wave,(wave.shape[0],1,-1,1))
                # output = self.sess.run(enhanced,{x_test:wave})
                # output = output.flatten()
                # output = output[:y.shape[0]]
                # print(np.max(output),np.min(output))
                # wav.write(test_path+"enhanced/"+name.split('/')[-1],16000,np.int16(output*32767))

                pad_num = OVERLAP * np.ceil((y.shape[0] - FRAMELENGTH)*1. / OVERLAP) + FRAMELENGTH
                temp = np.zeros(int(pad_num))
                out_temp = np.zeros(int(pad_num))
                sample_weight = np.zeros(int(pad_num))
                temp[:y.shape[0]] = y
                signals = slice_signal(temp, FRAMELENGTH, OVERLAP)
                wave = (2./65535.) * (signals.astype(np.float32)-32767) + 1.
                print(np.max(wave),np.min(wave))
                for idx, w in enumerate(wave):
                    w = np.reshape(w,(1,1,-1,1))
                    o =  self.sess.run(enhanced,{x_test:w})
                    out_temp[idx*OVERLAP:idx*OVERLAP+FRAMELENGTH] += o[0,0,:,0]
                    sample_weight[idx*OVERLAP:idx*OVERLAP+FRAMELENGTH] += 1
                out_temp = out_temp/sample_weight
                output = out_temp[:y.shape[0]]
                print(np.max(output),np.min(output))
                temp_name = name.split('/')
                wav.write(os.path.join(test_path,"enhanced",temp_name[-3],temp_name[-2],temp_name[-1]),16000,np.int16(output*32767))
                # wav.write(test_path+"enhanced/"+name.split('/')[-1],16000,np.int16(output*32767))
