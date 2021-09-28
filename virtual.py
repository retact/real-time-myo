import multiprocessing as mp
import numpy as np
import ctypes as c
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

path = ''
myo1 = pd.read_csv(path+'section1/above/2021-09-13T18_31_16.434477_emg.csv')
myo2 = pd.read_csv(path+'section1/under/2021-09-13T18_31_18.590345_emg.csv')
window_size = 1000
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
    while True:
    #print(emg_data)
    #print()
        time.sleep(0.5)

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

  #label, = ax_label.plot(np.zeros(100),label_data.reshape(-1))
  #label_text = ax_label.text(0.5,0.8,'Label Name',fontsize=20,transform=ax_label.transAxes,horizontalalignment='center')

    def update(i):

        for i, signal in enumerate(signals):
            signal.set_ydata(emg_data[:,i])

      # label.set_ydata(label_data)
      # label.set_xdata(label_time)
      # label_text.set_text(str(label_data.reshape(-1)[-1])+' : '+label_names[label_data.reshape(-1)[-1]])
      # ax_label.set_xlim(label_time[-1]-3,label_time[-1])

    ani = FuncAnimation(fig,update,2000,interval=100)
    plt.show()

if __name__=='__main__':


    try:
        with mp.Manager() as manager:

            emg_data = CreateArray(window_size,16)
        
            armband1 = mp.Process(target=virtual_myo1,args=(emg_data,))
            armband2 = mp.Process(target=virtual_myo2,args=(emg_data,))
            classifier = mp.Process(target=classifiers,args=(emg_data,))
            viewer = mp.Process(target=viewers,args=(emg_data,))
    
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
