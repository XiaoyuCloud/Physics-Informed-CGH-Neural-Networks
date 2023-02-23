import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as io
from model_PIUCGHNN_big import PIUCGHModel
from keras.models import Model
from sklearn.metrics import mean_squared_error
from diffractsim.general import *

M=128*1
N=64*1
test_input_mat = h5py.File('train/train_input.mat','r')
test_input=np.transpose(test_input_mat['train_input'])
test_input_4D=test_input.reshape(test_input.shape[0],N,M,1)
test_input_4D.shape

HoloAmp_mat = h5py.File('train/HoloAmp.mat')
#HoloAmp_mat = h5py.File('HoloAmp_multi_object_multi_depth.mat')
HoloAmp=np.transpose(HoloAmp_mat['HoloAmp'])
HoloAmp_4D=HoloAmp.reshape(HoloAmp.shape[0],N,M,1)

input_true=holoToRgb(HoloAmp_4D)
mse2=mean_squared_error(test_input_4D[0,::,::,0], input_true[0,::,::,0])