# -*- coding: utf-8 -*-
"""
Created on Thu 23:33 04/11/2019

@author: Cheng Yu
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from scipy.io import wavfile
import tensorflow as tf
import pdb
import scipy.io
import librosa
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Your GPU number, default = 0
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
KTF.set_session(session)
import time  
import numpy as np
import numpy.matlib
import random
random.seed(999)

Num_traindata= 270
epoch=60
batch_size=1
#valid_portion = 300

def creatdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            #pdb.set_trace()
            if filepath.split('/')[-1][-4:] == '.wav': #and filepath.split('/')[-2] != '-10db':
                file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.     
           
def data_generator(noisy_list, clean_path, shuffle = "False"):
    index=0
    while True:     
         #random.shuffle(noisy_list)
         rate, noisy = wavfile.read(noisy_list[index])
         noisy=noisy.astype('float')         
         if len(noisy.shape)==2:
             noisy=(noisy[:,0]+noisy[:,1])/2       
    
         noisy=noisy/np.max(abs(noisy))
         noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
         
         rate, clean = wavfile.read(clean_path+noisy_list[index].split('/')[-1])
         clean=clean.astype('float')  
         if len(clean.shape)==2:
             clean=(clean[:,0]+clean[:,1])/2

         clean=clean/np.max(abs(clean))         
         clean=np.reshape(clean,(1,np.shape(clean)[0],1))
         
         index += 1
         if index == len(noisy_list):
             index = 0
             if shuffle == "True":
                random.shuffle(noisy_list)
                       
         yield noisy, clean

def valid_generator(noisy_list, clean_path, shuffle = "False"):
    index=0
    while True:
         #random.shuffle(noisy_list)
         rate, noisy = wavfile.read(noisy_list[index])
         noisy=noisy.astype('float')
         if len(noisy.shape)==2:
             noisy=(noisy[:,0]+noisy[:,1])/2

         noisy=noisy/np.max(abs(noisy))
         noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))

         rate, clean = wavfile.read(clean_path+noisy_list[index].split('/')[-1])
         clean=clean.astype('float')
         if len(clean.shape)==2:
             clean=(clean[:,0]+clean[:,1])/2

         clean=clean/np.max(abs(clean))
         clean=np.reshape(clean,(1,np.shape(clean)[0],1))

         index += 1
         if index == len(noisy_list):
             index = 0
             if shuffle == "True":
                random.shuffle(noisy_list)
                       
         yield noisy, clean 

######################### Training data #########################
Train_Bone_lists = get_filepaths("/mnt/md2/user_khhung/bone_conduct/wavfile/Train/Bone")#training noisy set
Train_Air_paths = "/mnt/md2/user_khhung/bone_conduct/wavfile/Train/Air/"

random.shuffle(Train_Bone_lists)

idx = int(len(Train_Bone_lists)*0.95)

Train_lists = Train_Bone_lists[0:idx]
Valid_lists = Train_Bone_lists[idx:]


steps_per_epoch = (Num_traindata)//batch_size
######################### Test_set #########################
Test_Bone_lists = get_filepaths("/mnt/md2/user_khhung/bone_conduct/wavfile/Test/Bone")#[0:25] #testing noisy set
#Test_Clean_lists = get_filepaths("/mnt/md2/Corpora/TMHINT/Testing/clean")
#Test_Clean_paths = "/mnt/md2/Corpora/TMHINT/Testing/clean/" # testing clean set 
#pdb.set_trace()
Num_testdata=len(Valid_lists)

    
start_time = time.time()

print ('model building...')

model = Sequential()

#model.add(Convolution1D(1, 257,  border_mode='same', bias=False, input_shape=(None,1)))

model.add(Convolution1D(1, 35, border_mode='same', bias=False, input_shape=(None,1)))
#model.add(BatchNormalization(mode=2,axis=-1))
#model.add(LeakyReLU())

model.add(Convolution1D(15, 35,  border_mode='same', bias=False))
model.add(BatchNormalization(mode=2,axis=-1))

#model.add(LeakyReLU())

model.add(Convolution1D(15, 35,  border_mode='same', bias=False))
model.add(BatchNormalization(mode=2,axis=-1))

'''
model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(15, 35,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())
'''
model.add(Convolution1D(1, 35,  border_mode='same', bias=False))
model.add(Activation('tanh'))
model.summary()
#pdb.set_trace()
model.compile(loss='mse', optimizer='adam')
    
with open('FCN_b2a_V4.json','w') as f:    # save the model
    f.write(model.to_json()) 
checkpointer = ModelCheckpoint(filepath='FCN_b2a_V4.hdf5', verbose=1, save_best_only=True, mode='min')  
#pdb.set_trace()
'''
layer_void = model.layers
weights = layer_void[0].get_weights()
#filter_out = layer_void[-2].get_weights()
#print(weights[0][55//2])
#weights[0][55//2] = 0   

#for layer in model.layers:
    #print(layer.trainable)
#layer_void[0].trainable = False
#pdb.set_trace()
#print(weights[0][(55//2)-18:(55//2)+18])
weights[0][:] = 1/257
weights[0][(257//2)-64:(257//2)+64] = 0
#filter_out[0][(512//2)-128:(512//2)+128] = 0
#weights[0][:] = 0
#pdb.set_trace()
#print(weights[0][(512//2)-128:(512//2)+128])
layer_void[0].trainable = False
#layer_void[-2].trainable = False
'''
print('training...')

g1 = data_generator(Train_lists, Train_Air_paths, shuffle = "True")
g2 = valid_generator(Valid_lists, Train_Air_paths, shuffle = "False")                					

#for e in range(epoch):
    #for dn in range(Num_traindata):
         #step = ["epoch:", e, ", process:", dn] 
         #print(step)
         #weights[0][(55//2)-6:(55//2)-3+6] = 0
hist=model.fit_generator(g1,    
                         samples_per_epoch=Num_traindata, 
                         nb_epoch=epoch, 
                         verbose=1,
                         validation_data=g2,
                         nb_val_samples=Num_testdata,
                         max_q_size=20, 
                         nb_worker=3,
                         pickle_safe=True,
                         callbacks=[checkpointer]
                         )                                   
#tf.reset_default_graph()
#print(weights[0][55//2])


tStart = time.time()

print('load model')
MdNamePath='FCN_b2a_V4' #the model path
with open(MdNamePath+'.json') as f:
    model = model_from_json(f.read());
        
model.load_weights(MdNamePath+'.hdf5');
model.summary()
pdb.set_trace()
print(K.floatx())
print('testing...')

for path in Test_Bone_lists: # Ex: /mnt/Nas/Corpus/TMHINT/Testing/Noisy/car_noise_idle_noise_60_mph/b4/1dB/TMHINT_12_10.wav
    if path.split("/")[-1][-4:] == ".wav":
        #pdb.set_trace()
        S=path.split('/') 
        noise=S[-4]
        speaker=S[-3]
        dB=S[-2]    
        wave_name=S[-1]
    
        rate, noisy = wavfile.read(path)
        noisy=noisy.astype('float32')
        if len(noisy.shape)==2:
            noisy=(noisy[:,0]+noisy[:,1])/2
    
        noisy=noisy/np.max(abs(noisy))
        #for t in range(30):
        #noisy = Gau_Gen(noisy)
        #pdb.set_trace()
        noisy=np.reshape(noisy,(1,noisy.shape[0],1))
        #pdb.set_trace() 
        enhanced=np.squeeze(model.predict(noisy, verbose=0, batch_size=batch_size))
        enhanced=enhanced/np.max(abs(enhanced))
        enhanced=enhanced.astype('float32')
        #    creatdir(os.path.join("Gaussian_noisy", noise, speaker, dB))
        #    librosa.output.write_wav(os.path.join("Gaussian_noisy", noise, speaker, dB, str(t)+"_"+wave_name), noisy, 16000)
        creatdir(os.path.join("FCN_b2a_MAE_wav_V4", noise, speaker, dB))
        librosa.output.write_wav(os.path.join("FCN_b2a_MAE_wav_V4", noise, speaker, dB, wave_name), enhanced, 16000)
tEnd = time.time()
print "It cost %f sec" % (tEnd - tStart)

# plotting the learning curve
TrainERR=hist.history['loss']
ValidERR=hist.history['val_loss']
print('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
print('drawing the training process...')
plt.figure(2)
plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
plt.xlim([1,epoch])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig('FCN_b2a_mae_Learning_curve_V4.png', dpi=150)


end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))

