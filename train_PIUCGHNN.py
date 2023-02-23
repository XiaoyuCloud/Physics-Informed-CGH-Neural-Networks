import matplotlib.pyplot as plt
import h5py
from model_PIUCGHNN_big import PIUCGHModel
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from diffractsim.general import *

def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()
M=64*1
N=64*1
train_input_mat = h5py.File('train/train_input.mat')
#train_input_mat = h5py.File('train_input_multi_object_multi_depth.mat')
train_input=np.transpose(train_input_mat['train_input'])
train_input_4D=train_input.reshape(train_input.shape[0],N,M,1)

print(train_input_4D.shape)

HoloAmp_mat = h5py.File('train/HoloAmp.mat')
#HoloAmp_mat = h5py.File('HoloAmp_multi_object_multi_depth.mat')
HoloAmp=np.transpose(HoloAmp_mat['HoloAmp'])
HoloAmp_4D=HoloAmp.reshape(HoloAmp.shape[0],N,M,1)

model = PIUCGHModel()
print(model.summary())

try:
    model.load_weights("SaveModel/DLH.h5")
    #model.load_weights("SaveModel/best_weights.h5")
    print("加载模型成功!继续训练模型")
except:
   print("加载模型失败!开始训练一个新模型")

def MSE_PIUNN(y_true,y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='mean_squared_error',optimizer=adam, metrics=['accuracy'])
model.compile(loss=MSE_PIUNN,optimizer=adam, metrics=['accuracy'])

#创建一个权重文件保存文件夹logs
log_dir = "SaveModel/logs/"
#记录所有训练过程，每隔一定步数记录最大值
tensorboard = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + "best_weights.h5",
                                 monitor="loss",
                                 mode='min',
                                 save_weights_only=True,
                                 save_best_only=True, 
                                 verbose=1,
                                 period=1)
callback_lists=[tensorboard,checkpoint]
#train_history=model.fit(train_input_4D, train_input_4D,validation_split=0.2,epochs=3000, batch_size=1, callbacks=callback_lists, verbose=1)
train_history=model.fit(train_input_4D, train_input_4D,validation_split=0,epochs=5, batch_size=1, verbose=1)

def show_train_history(train_history,train):
    plt.plot(train_history.history[train])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

show_train_history(train_history,'loss')
show_train_history(train_history,'acc')
model.save_weights("SaveModel/DLH.h5")
print("Saved model to disk")

