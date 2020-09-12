# -*- coding: utf-8 -*-
"""
Created on Thu 23:33 04/11/2019

@author: Cheng Yu
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, model_from_json, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Convolution1D
from keras.engine.topology import Merge
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.visualize_util import plot
from scipy.io import wavfile
import tensorflow as tf
import pdb
import scipy.io
import librosa
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" #Your GPU number, default = 0
import keras.backend.tensorflow_backend as KTF
#config = tf.ConfigProto()
config = tf.ConfigProto(device_count={"CPU": 1},
                inter_op_parallelism_threads = 5,
                intra_op_parallelism_threads = 5,
                log_device_placement=False)

config.gpu_options.per_process_gpu_memory_fraction = 0.5
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
           
def data_generator(noisy_list, bone_path, clean_path, shuffle = "False"):
    index=0
    while True:     
         #random.shuffle(noisy_list)
         rate, noisy = wavfile.read(noisy_list[index])
         noisy=noisy.astype('float')         
         if len(noisy.shape)==2:
             noisy=(noisy[:,0]+noisy[:,1])/2       
    
         noisy=noisy/np.max(abs(noisy))
         noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
         
         rate, bone = wavfile.read(bone_path+noisy_list[index].split('/')[-1][0:4]+".wav")
         bone=bone.astype('float')
         if len(bone.shape)==2:
             bone=(bone[:,0]+bone[:,1])/2

         bone=bone/np.max(abs(bone))
         bone=np.reshape(bone,(1,np.shape(bone)[0],1))
         
         #noisy = np.concatenate((noisy,bone), axis = -1)
           
         rate, clean = wavfile.read(clean_path+noisy_list[index].split('/')[-1][0:4]+".wav")
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
                       
         yield [bone, noisy], clean

def valid_generator(noisy_list, bone_path, clean_path, shuffle = "False"):
    index=0
    while True:
         #random.shuffle(noisy_list)
         rate, noisy = wavfile.read(noisy_list[index])
         noisy=noisy.astype('float')
         if len(noisy.shape)==2:
             noisy=(noisy[:,0]+noisy[:,1])/2
         
         noisy=noisy/np.max(abs(noisy))
         noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))

         rate, bone = wavfile.read(bone_path+noisy_list[index].split('/')[-1][0:4]+".wav")
         bone=bone.astype('float')
         if len(bone.shape)==2:
             bone=(bone[:,0]+bone[:,1])/2

         bone=bone/np.max(abs(bone))
         bone=np.reshape(bone,(1,np.shape(bone)[0],1))

         #noisy = np.concatenate((noisy,bone), axis = -1)

         rate, clean = wavfile.read(clean_path+noisy_list[index].split('/')[-1][0:4]+".wav")
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
                       
         yield [bone, noisy], clean 

######################### Training data #########################
Train_Bone_paths = "/mnt/md2/user_khhung/bone_conduct/wavfile/Train/Bone/"#training noisy set
Train_Noisy_lists = get_filepaths("/mnt/md2/user_khhung/bone_conduct/wavfile/Train/Noisy")
Train_Air_paths = "/mnt/md2/user_khhung/bone_conduct/wavfile/Train/Air/"

random.shuffle(Train_Noisy_lists)

idx = int(len(Train_Noisy_lists)*0.95)

Train_lists = Train_Noisy_lists[0:idx]
Valid_lists = Train_Noisy_lists[idx:]
Num_traindata = len(Train_lists)

steps_per_epoch = (Num_traindata)//batch_size
######################### Test_set #########################
Test_Bone_paths = "/mnt/md2/user_khhung/bone_conduct/wavfile/Test/Bone/"#[0:25] #testing noisy set
Test_Noisy_lists = get_filepaths("/mnt/md2/user_khhung/bone_conduct/wavfile/Test/Noisy/")
#Test_Clean_lists = get_filepaths("/mnt/md2/Corpora/TMHINT/Testing/clean")
#Test_Clean_paths = "/mnt/md2/Corpora/TMHINT/Testing/clean/" # testing clean set 
#pdb.set_trace()
Num_testdata=len(Valid_lists)

start_time = time.time()

print ('model building...')

print('load bone')
MdNamePath_b='FCN_b2a_retrain' #the model path
with open(MdNamePath_b+'.json') as f:
    bone = model_from_json(f.read());

bone.load_weights(MdNamePath_b+'.hdf5');

layer_bone = bone.layers

for idx in range(len(layer_bone)):
    
    layer_bone[idx].trainable = False

bone.summary()

'''
###no pretrain bone model
bone=Sequential()

#model.add(Convolution1D(1, 257,  border_mode='same', bias=False, input_shape=(None,1)))

bone.add(Convolution1D(1, 35, border_mode='same', bias=False, input_shape=(None,1)))
#model.add(BatchNormalization(mode=2,axis=-1))
#model.add(LeakyReLU())

bone.add(Convolution1D(15, 35,  border_mode='same', bias=False))
bone.add(BatchNormalization(mode=2,axis=-1))

#model.add(LeakyReLU())

bone.add(Convolution1D(15, 35,  border_mode='same', bias=False))
bone.add(BatchNormalization(mode=2,axis=-1))
bone.add(Convolution1D(1, 35,  border_mode='same', bias=False))
bone.add(Activation('tanh'))
bone.compile(loss='mse', optimizer='adam')
bone.summary()
'''

print('load air')
MdNamePath_a='FCN_a2a_V1' #the model path
with open(MdNamePath_a+'.json') as f:
    air = model_from_json(f.read());

air.load_weights(MdNamePath_a+'.hdf5');
layer_air = air.layers

for idx in range(len(layer_air)):

    layer_air[idx].trainable = False

air.summary()

model_b = Sequential()

#model_b.add(Input(shape=(None,1)))

model_b.add(bone)

model_a = Sequential()

#model_a.add(Input(shape=(None,1)))

model_a.add(air)

model = Sequential()

model.add(Merge([model_b, model_a], mode='concat', concat_axis=-1))

model.add(Convolution1D(15, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
'''
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())

model.add(Convolution1D(30, 55,  border_mode='same'))
model.add(BatchNormalization(mode=2,axis=-1))
model.add(LeakyReLU())
'''
model.add(Convolution1D(1, 55,  border_mode='same'))
model.add(Activation('tanh'))

#pdb.set_trace()

'''
model1 = Sequential()
model1.add(Convolution1D(30, 55,  border_mode='same', input_shape=(None,1)))

model2 = Sequential()
model2.add(Convolution1D(30, 55,  border_mode='same', input_shape=(None,1)))

merged_model = Sequential()
merged_model.add(Merge([model1, model2], mode='concat', concat_axis=-1))
merged_model.add(Convolution1D(30, 55,  border_mode='same'))
merged_model.add(BatchNormalization(mode=2,axis=-1))
'''

model.summary()
#pdb.set_trace()
model.compile(loss='mse', optimizer='adam')

with open('FCN_late_pretrain.json','w') as f:    # save the model
    f.write(model.to_json()) 
checkpointer = ModelCheckpoint(filepath='FCN_late_pretrain.hdf5', verbose=1, save_best_only=True, mode='min')  
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

g1 = data_generator(Train_lists, Train_Bone_paths, Train_Air_paths, shuffle = "True")
g2 = valid_generator(Valid_lists, Train_Bone_paths, Train_Air_paths, shuffle = "False")                					

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
                         max_q_size=2, 
                         nb_worker=5,
                         pickle_safe=True,
                         callbacks=[checkpointer]
                         )                                   
#tf.reset_default_graph()
#print(weights[0][55//2])


tStart = time.time()

print('load model')
MdNamePath='FCN_late_pretrain' #the model path
with open(MdNamePath+'.json') as f:
    model = model_from_json(f.read());
        
model.load_weights(MdNamePath+'.hdf5');
model.summary()
pdb.set_trace()
print(K.floatx())
print('testing...')

for path in Test_Noisy_lists: # Ex: /mnt/Nas/Corpus/TMHINT/Testing/Noisy/car_noise_idle_noise_60_mph/b4/1dB/TMHINT_12_10.wav
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
        noisy=np.reshape(noisy,(1,noisy.shape[0],1))

        rate, bone = wavfile.read(Test_Bone_paths+path.split('/')[-1][0:4]+".wav")
        bone=bone.astype('float')
        if len(bone.shape)==2:
            bone=(bone[:,0]+bone[:,1])/2

        bone=bone/np.max(abs(bone))
        bone=np.reshape(bone,(1,np.shape(bone)[0],1))
        #noisy = np.concatenate((noisy,bone), axis = -1)

        enhanced=np.squeeze(model.predict([bone, noisy], verbose=0, batch_size=batch_size))
        enhanced=enhanced/np.max(abs(enhanced))
        enhanced=enhanced.astype('float32')
        #    creatdir(os.path.join("Gaussian_noisy", noise, speaker, dB))
        #    librosa.output.write_wav(os.path.join("Gaussian_noisy", noise, speaker, dB, str(t)+"_"+wave_name), noisy, 16000)
        creatdir(os.path.join("FCN_late_pretrain_MSE_wav", noise, speaker, dB))
        librosa.output.write_wav(os.path.join("FCN_late_pretrain_MSE_wav", noise, speaker, dB, wave_name), enhanced, 16000)
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
plt.savefig('FCN_late_ab_pretrain_Learning_curve.png', dpi=150)


end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))

