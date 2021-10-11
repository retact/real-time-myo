import multiprocessing as mp
import numpy as np
import ctypes as c
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Sequential, load_model

window_size = 200

#CPU only
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

label_names = ['Rest',
              'Thumbs up',
              'Hand sign 2',
              'Hand sign 3',
              'Hand sign 4',
              'Hand sign 5',
              'Grip',
              'Point',
              'Finger close',
]

import math
def valid_convolve(xx, size):
    b = np.ones(size)/size
    xx_mean = np.convolve(xx, b, mode="same")
    n_conv = math.ceil(size/2)

    # 補正部分
    xx_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= size/(i+n_conv)
        xx_mean[-i] *= size/(i + n_conv - (size % 2)) 
# size%2は奇数偶数での違いに対応するため

    return xx_mean


def preprocess(x_data,len_window=window_size):

  #Preprocess1 正規化
  #x_data1 = (x_data-x_data.mean(axis=1).reshape(x_data.shape[0],1,x_data.shape[2]))/x_data.std(axis=1).reshape(x_data.shape[0],1,x_data.shape[2])

    #Preprocess2 整流
    x_data2 = np.abs(x_data)

  #Preprocess3 移動平均フィルタ
    x_data3 = np.zeros_like(x_data2)
    for i in range(x_data2.shape[0]):
        for j in range(x_data2.shape[2]):
            x_data3[i,:,j] = valid_convolve(x_data2[i,:,j],len_window)

      # #データ可視化
      # fig = plt.figure(figsize=(10,8))
      # plt.subplots_adjust(wspace=0.0, hspace=0.3)
      # ax1 = fig.add_subplot(211)
      # ax2 = fig.add_subplot(212)

      # ax1.plot(x_data[i,:,:])
      # ax1.set_title('Raw')
      # ax1.set_xlim(0,len_window)
      # ax1.set_ylim(-100,100)
      # ax2.plot(x_data3[i,:,:])
      # ax2.set_title('Processed')
      # ax2.set_xlabel('number of data')
      # ax2.set_xlim(0,len_window)
      # #ax2.set_ylim(0,1.2)
      # plt.show()

    return x_data3

path = '../datasets/20210913_fukano/'
#myo1 = pd.read_csv(path+'section1/above/2021-09-13T18_31_16.434477_emg.csv')
#myo2 = pd.read_csv(path+'section1/under/2021-09-13T18_31_18.590345_emg.csv')

myo1 = pd.read_csv(path+'section2/above/2021-09-13T17_04_26.500235_emg.csv')
myo2 = pd.read_csv(path+'section2/under/2021-09-13T17_04_29.167477_emg.csv')

if myo1['timestamp'][0] <= myo2['timestamp'][0]:
  myo2['timestamp'] -= myo1['timestamp'][0]
  myo1['timestamp'] -= myo1['timestamp'][0]
else:
  myo1['timestamp'] -= myo2['timestamp'][0]
  myo2['timestamp'] -= myo2['timestamp'][0]

def CreateArray(n,m):
  mp_arr=mp.Array('i',n*m)
  arr = np.frombuffer(mp_arr.get_obj(),c.c_int)  # mp_arr and arr share the same memory
  # make it two-dimensional
  #print('np array len=',len(arr))
  b = arr.reshape((n, m))  # b and arr share the same memory
  return b

def virtual_myo1(emg_data):
  delay = myo1['timestamp'][0]
  time.sleep(delay)
  for i in range(len(myo1['timestamp'])-1):
    emg_data[:-1,:8] = emg_data[1:,:8]
    emg_data[-1,:8] = myo1.loc[i,'emg1':'emg8']
    delay = myo1['timestamp'][i+1] - myo1['timestamp'][i]
    time.sleep(delay)

def virtual_myo2(emg_data):
  delay = myo2['timestamp'][0]
  time.sleep(delay)
  for i in range(len(myo2['timestamp'])-1):
    emg_data[:-1,8:] = emg_data[1:,8:]
    emg_data[-1,8:] = myo2.loc[i,'emg1':'emg8']
    delay = myo2['timestamp'][i+1] - myo2['timestamp'][i]
    time.sleep(delay)

def classifiers(emg_data):
    cnn_model = load_model('models/fukano_8label_windowsize200.h5')
    #cnn_model.compile(loss='categorical_crossentropy',optimizer='adam')
    cnn_model.summary()
    program_start_time = time.time()
    while True:
        preprocessed = preprocess(emg_data.reshape(1,window_size,16))
        emg_data_preprocessed = preprocessed

        start_time = time.time()
        predicts = cnn_model.predict(emg_data_preprocessed.reshape(1,window_size,16,1))
        label = np.argmax(predicts)
        end_time = time.time()

        print('\nTime Required : %f'%(end_time-start_time))
        print('Predicted Label : %d'%(label),label_names[label])


        emg_data_preprocessed[:-1] = emg_data_preprocessed[1:]
        emg_data_preprocessed[-1] = label
        label_data[:-1] = label_data[1:]
        label_data[-1] = label
        label_time[:-1] = label_time[1:]
        label_time[-1] = time.time()-program_start_time

def viewers(emg_data):
  fig = plt.figure(figsize=(10,5))
  fig.canvas.set_window_title('Viewer')
  fig.subplots_adjust(hspace=0.6,wspace=0.6)

  ax_myo1 = fig.add_subplot(221)
  ax_myo1.set(xlim=[0,window_size],ylim=[-128,127])
  ax_myo1.set_title('Myo Armband 1',fontsize=15)
  ax_myo1.set_ylabel('EMG signals',fontsize=12)
  ax_myo1.set_xlabel('Number of Data',fontsize=12)

  ax_myo2 = fig.add_subplot(223)
  ax_myo2.set(xlim=[0,window_size],ylim=[-128,127])
  ax_myo2.set_title('Myo Armband 2',fontsize=15)
  ax_myo2.set_ylabel('EMG signals',fontsize=12)
  ax_myo2.set_xlabel('Number of Data',fontsize=12)

  ax_label = fig.add_subplot(122)
  ax_label.set_xlabel('Time',fontsize=12)
  ax_label.set_ylabel('Label',fontsize=12)
  ax_label.set(ylim=[0,12])
  

  signals = []
  for i in range(8):
      signal, = ax_myo1.plot(np.arange(window_size),np.zeros(window_size),alpha=0.4)
      signals.append(signal)
  for i in range(8):
      signal, = ax_myo2.plot(np.arange(window_size),np.zeros(window_size),alpha=0.4)
      signals.append(signal)

  label, = ax_label.plot(np.zeros(500),label_data.reshape(-1))
  label_text = ax_label.text(0.5,0.8,'Label Name',fontsize=20,transform=ax_label.transAxes,horizontalalignment='center')

  def update(i):

      for i, signal in enumerate(signals):
          signal.set_ydata(emg_data[:,i])

      label.set_ydata(label_data)
      label.set_xdata(label_time)
      label_text.set_text(str(label_data.reshape(-1)[-1])+' : '+label_names[label_data.reshape(-1)[-1]])
      ax_label.set_xlim(label_time[-1]-2,label_time[-1])

  ani = FuncAnimation(fig,update,interval=100)
  plt.show()

if __name__=='__main__':

  try:
    with mp.Manager() as manager:

        emg_data = CreateArray(window_size,16)
        label_data = CreateArray(500,1)
        label_time = CreateArray(500,1)
        
        armband1 = mp.Process(target=virtual_myo1,args=(emg_data,))
        armband2 = mp.Process(target=virtual_myo2,args=(emg_data,))
        classifier = mp.Process(target=classifiers,args=(emg_data,))
        viewer = mp.Process(target=viewers,args=(emg_data,))

        armband2.start()
        armband1.start()
        classifier.start()
        viewer.start()

        armband2.join()
        armband1.join()
        classifier.join()
        viewer.join()

  except KeyboardInterrupt:
    pass
