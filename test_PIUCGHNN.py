import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as io
from model_PIUCGHNN_big import PIUCGHModel
from keras.models import Model
from sklearn.metrics import mean_squared_error
from diffractsim.general import *

model = PIUCGHModel()

model.load_weights("SaveModel/DLHsingleNum0.0013.h5")
#model.load_weights("SaveModel/best_weights.h5")
print("加载模型成功!")
M=64*1
N=64*1
test_input_mat = h5py.File('train/train_input.mat','r')
test_input=np.transpose(test_input_mat['train_input'])
test_input_4D=test_input.reshape(test_input.shape[0],N,M,1)
test_input_4D.shape

# prediction
input_pred = model.predict(test_input_4D, batch_size=1)
mse=mean_squared_error(test_input_4D[0,::,::,0], input_pred[0,::,::,0])
CGH_layer_model = Model(inputs=model.input,outputs=model.get_layer('conv2d_47').output)
CGH = CGH_layer_model.predict(test_input_4D)
print (CGH.shape)

# plot results
plt.figure()
for i in range(5):
    plt.subplot(5, 4, i*4 + 1)
    plt.imshow(test_input_4D[i, :].squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(5, 4, i*4 + 2)
    plt.imshow(CGH[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(5, 4, i*4 + 3)
    plt.imshow(input_pred[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.show()

io.savemat('test/momd/test_output_pred.mat', {'HoloAmp': CGH})

