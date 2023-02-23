import diffractsim
from pathlib import Path
from PIL import Image
import numpy as np
import h5py

diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, mm, nm, cm
from diffractsim.general import *
import torch
from sklearn.metrics import mean_squared_error
from keras import backend as K

# img = Image.open(Path("./apertures/holo.bmp"))
# holo=np.asarray(img) / 255.0
#img.show()
holo_mat = h5py.File('train/HoloAmp.mat','r')
holo=np.transpose(holo_mat['HoloAmp'])
holo_4D=holo.reshape(holo.shape[0],64,64,1)
holo=holo_4D[0,::,::,0]
pitch=10.8e-3
M=64
N=64
EX=M*pitch
EY=N*pitch

test_input_mat = h5py.File('train/train_input.mat','r')
test_input=np.transpose(test_input_mat['train_input'])
test_input_4D=test_input.reshape(test_input.shape[0],N,M,1)

F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=EX * mm, extent_y=EY * mm, Nx=M, Ny=N
)

# F.add_aperture_from_image(
#     "./apertures/holo.bmp", image_size=(20.736 * mm, 11.664 * mm)
# )
#F.add_aperture_from_array(holo, image_size=(EX * mm, EY * mm))
F.add_aperture_from_arrayTF(torch.from_numpy(holo))

rgb = F.compute_colors_atTF(-50*mm)
#rgbGray=RgbToGray(rgb)
rgbGray = tf.cast(rgb, tf.float32)
rgbGrayNor= noramlizationTF(rgbGray)
# #rgbGrayNor4D=rgbGrayNor.reshape(1,64,64,1)
rgbGrayNor4D = tf.reshape(rgbGrayNor,[1, 64, 64, 1])
rgbGrayNor=rgbGrayNor4D[0,::,::,0]
rgbGrayNor=K.eval(rgbGrayNor)
mse=mean_squared_error(test_input_4D[0,::,::,0], rgbGrayNor)
F.plot(rgbGrayNor, figsize=(0.9, 0.9),xlim=[-EX/2, EX/2], ylim=[-EY/2, EY/2])
