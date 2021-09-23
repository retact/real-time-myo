
# 
# Licensed under the MIT license. See the LICENSE file for details.
#

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ctypes as c
import sys
import time
from myoraw import MyoRaw, DataCategory, EMGMode


def CreateArray(n, m):
    mp_arr = mp.Array('i', n*m)
    arr = np.frombuffer(mp_arr.get_obj(), c.c_int)
    b = arr.reshape((n, m))  # b and arr share the same memory
    return b


def proc_emg1(timestamp, emg, moving, characteristic_num, times=[]):
    # print('Myo1 wrote')
    # print(emg)
    # print(time.time())
    emg_data[:-1, :8] = emg_data[1:, :8]
    emg_data[-1, :8] = emg


def proc_emg2(timestamp, emg, moving, characteristic_num, times=[]):
    # print('Myo2 wrote')
    # print(emg)
    # print(time.time())
    emg_data[:-1, 8:] = emg_data[1:, 8:]
    emg_data[-1, 8:] = emg


def viewer(emg_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    fig.subplots_adjust(hspace=0.6)
    ax1.set(xlim=[0, window_size], ylim=[-128, 127])
    # ax2.set(xlim=[0,window_size],ylim=[-128,127])
    ax1.set_title('Myo Armband 1', fontsize=15)
    # ax2.set_title('Myo Armband 2',fontsize=15)
    ax1.set_ylabel('EMG signals', fontsize=12)
    # ax2.set_ylabel('EMG signals',fontsize=12)
    ax1.set_xlabel('Number of Data', fontsize=12)
    # ax2.set_xlabel('Number of Data',fontsize=12)
    ax2 = fig.add_subplot(212, projection="3d")

    signals = []
    dsignals = []

    for i in range(8):
        signal, = ax1.plot(np.arange(window_size), np.zeros(window_size), alpha=0.4)
        signals.append(signal)
        dsignal, = ax2.plot(np.arange(windowsize), np.full_like(window_size, i+1),
                            np.zeros(window_size))
    #for i in range(8):
    #    signal, = ax2.plot(np.arange(window_size),np.zeros(window_size),alpha=0.4)
    #    signals.append(signal)

    def update(i):
        for count_1, signal in enumerate(signals):
            signal.set_ydata(emg_data[:, count_1])
        for count_2, 3dsignal in enumrate(3dsignals):
            dsignal.set_ydata(np.full_like(window_size, count_2+1))
            dsignal.set_3dproperties(emg_data[:, count_2])

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()

m1 = MyoRaw('/dev/ttyACM0')
time.sleep(0.2)
#m2 = MyoRaw('/dev/ttyACM1')

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

        window_size = 1000
        #emg_data = CreateArray(window_size, 16)
        emg_data = CreateArray(window_size, 8)

        armband1 = mp.Process(target=myoarmband1, args=(emg_data,))
        #armband2 = mp.Process(target=myoarmband2,args=(emg_data,))
        viewer = mp.Process(target=viewer, args=(emg_data,))

        armband1.start()
        #armband2.start()
        viewer.start()

        armband1.join()
        #armband2.join()
        viewer.join()


except KeyboardInterrupt:
    pass
finally:
    m1.disconnect()
    #m2.disconnect()
    print("\nDisconnected")
