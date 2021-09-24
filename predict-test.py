# 
# Licensed under the MIT license. See the LICENSE file for details.
#

import math
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.animation import FuncAnimation
import ctypes as c
import sys
import time
import tkinter as tk
from myoraw import MyoRaw, DataCategory, EMGMode
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Sequential, load_model
import statistics

#CPU only
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Machine Learning model
window_size = 1000

label_names = ['Flex Index',
              'Extend Index',
              'Flex Middle',
              'Extend Middle',
              'Flex Ring',
              'Extend Ring',
              'Flex Little',
              'Extend Little',
              'Adduct Thumb',
              'Abduct Thumb',
              'Flex Thumb',
              'Extend Thumb',
              '12',
              '13',
              '14',
              '15',
              '16',
              '17',
              '18',
              '19',
              '20',
              '21',
              '22',
              '23',
              '24',
              '25',
              '26',
              '27',
              '28',
              '29',
              '30',
              '31',
              '32',
              '33',
              '34',
              '35',
              '36',
              '37',
              '38',
              '39',
              '40',
              '41',
              '42',
              '43',
              '44',
              '45',
              '46',
              '47',
              '48',
              '49',
              '50',
              '51',
]


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


def CreateArray(n,m):
    mp_arr=mp.Array('i',n*m)
    arr = np.frombuffer(mp_arr.get_obj(),c.c_int)
    b = arr.reshape((n, m))  # b and arr share the same memory
    return b


def proc_emg1(timestamp, emg, moving, characteristic_num, times=[]):
    #print('Myo1 wrote')
    #print(emg)
    #print(time.time())
    emg_data[:-1,:8] = emg_data[1:,:8]
    emg_data[-1,:8] = emg


def proc_emg2(timestamp, emg, moving, characteristic_num, times=[]):
    #print('Myo2 wrote')
    #print(emg)
    #print(time.time())
    emg_data[:-1,8:] = emg_data[1:,8:]
    emg_data[-1,8:] = emg

def viewer(emg_data,label_data,label_time):
    
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

    label, = ax_label.plot(np.zeros(100),label_data.reshape(-1))
    label_text = ax_label.text(0.5,0.8,'Label Name',fontsize=20,transform=ax_label.transAxes,horizontalalignment='center')

    def update(i):

        for i, signal in enumerate(signals):
            signal.set_ydata(emg_data[:,i])

        label.set_ydata(label_data)
        label.set_xdata(label_time)
        label_text.set_text(str(label_data.reshape(-1)[-1])+' : '+label_names[label_data.reshape(-1)[-1]])
        ax_label.set_xlim(label_time[-1]-3,label_time[-1])

    ani = FuncAnimation(fig,update,interval=100)
    plt.show()

def classifier(emg_data,label_data,label_time):
    # 何らかでlistでmodelを格納する
    cnn_model = load_model('fukano_model_12labels_1000.h5')
    #cnn_model.compile(loss='categorical_crossentropy',optimizer='adam')
    cnn_model.summary()
    program_start_time = time.time()
    while True:
        pre_times = []
        for i in range(100):
            preprocessed = preprocess(emg_data.reshape(1,window_size,16))
            emg_data_preprocessed = preprocessed

            start_time = time.time()
            label = np.argmax(cnn_model.predict(emg_data_preprocessed.reshape(1,window_size,16,1)))
            end_time = time.time()

            print('\nTime Required : %f'%(end_time-start_time))
            print('Predicted Label : %d'%(label),label_names[label])


            emg_data_preprocessed[:-1] = emg_data_preprocessed[1:]
            emg_data_preprocessed[-1] = label

            label_data[:-1] = label_data[1:]
            label_data[-1] = label
            label_time[:-1] = label_time[1:]
            label_time[-1] = time.time()-program_start_time
            pre_times.append(end_time-start_time)
        pre_mean = statistics.mean(pre_times)
        print("Average prediction time : ", pre_mean)


m1 = MyoRaw('/dev/ttyACM0')
time.sleep(0.2)
m2 = MyoRaw('/dev/ttyACM1')


def myoarmband1(emg_data):
    m1.add_handler(DataCategory.EMG, proc_emg1)
    m1.subscribe(EMGMode.RAW)
    m1.set_sleep_mode(1)
    m1.vibrate(1)
    while True:
        m1.run(1)


def myoarmband2(emg_data):
    m2.add_handler(DataCategory.EMG, proc_emg2)
    m2.subscribe(EMGMode.RAW)
    m2.set_sleep_mode(1)
    m2.vibrate(1)
    while True:
        m2.run(1)


try:
    with mp.Manager() as manager:

        emg_data = CreateArray(window_size,16)
        label_data = CreateArray(100,1)
        label_time = CreateArray(100,1)

        armband1 = mp.Process(target=myoarmband1,args=(emg_data,))
        armband2 = mp.Process(target=myoarmband2,args=(emg_data,))
        viewer = mp.Process(target=viewer,args=(emg_data,label_data,label_time))
        classifier = mp.Process(target=classifier,args=(emg_data,label_data,label_time))

        armband1.start()
        armband2.start()
        classifier.start()
        viewer.start()
        
        armband1.join()
        armband2.join()
        classifier.join()
        viewer.join()



except KeyboardInterrupt:
    pass
finally:
    m1.disconnect()
    m2.disconnect()
    plt.close()
    print("\nDisconnected")
