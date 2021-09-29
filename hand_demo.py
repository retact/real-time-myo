import Adafruit_ADS1x15
import Adafruit_PCA9685
import time
import numpy as np
#from scipy.signal import argrelmax
#from scipy.signal import argrelmin
import matplotlib.pyplot as plt

servo_num = np.array([0,3,7,8,11])
#servo minは120
#servo maxは675
servo_min = 300
servo_max = 600

adc = Adafruit_ADS1x15.ADS1115()
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)

class Base_finger:
    def __init__(self,finger_num,servo_num):
        self.old_state = 0.0
        self.new_state = 0.0
        self.finger_num = finger_num
        self.servo_num = servo_num
        
    def anti_shake(self):
        width = 0.3
        if self.new_state-width/2<self.old_state<self.new_state+width/2:
           self.new_state = self.old_state
           
    def initial_pos(self):
        if 0<=self.finger_num<=1:
            pwm.set_pwm(self.servo_num,0,int(servo_max-0*(servo_max-servo_min)))
        else:
            pwm.set_pwm(self.servo_num,0,int(0*(servo_max-servo_min)+servo_min))
            
    def move(self,state):
        self.new_state = state
        self.anti_shake()
        if 0<=self.finger_num<=1:
            if 0<=self.new_state<=1:
               pass
            elif self.new_state>1:
                self.new_state = 1.0
            else:
                self.new_state = 0.0
            pwm.set_pwm(self.servo_num,0,int(servo_max-self.new_state*(servo_max-servo_min)))
        else:
            if 0<=self.new_state<=1:
               pass
            elif self.new_state>1:
                self.new_state = 1.0
            else:
                self.new_state = 0.0
            pwm.set_pwm(self.servo_num,0,int(self.new_state*(servo_max-servo_min)+servo_min))

thumb = Base_finger(0,servo_num[0])
index = Base_finger(1,servo_num[1])
middle = Base_finger(2,servo_num[2])
ring = Base_finger(3,servo_num[3])
little = Base_finger(4,servo_num[4])

def open_hand():
    thumb.initial_pos()
    index.initial_pos()
    middle.initial_pos()
    ring.initial_pos()
    little.initial_pos()
    
def close_hand(state):
    thumb.move(state)
    index.move(state)
    middle.move(state)
    ring.move(state)
    little.move(state)
    
# while True:
#     open_hand()
#     time.sleep(1)
#     close_hand(1)
#     time.sleep(1)
#     open_hand()
#     time.sleep(1)
#     
#     thumb.move(1)
#     time.sleep(0.5)
#     index.move(0.9)
#     time.sleep(0.5)
#     middle.move(1)
#     time.sleep(0.5)
#     ring.move(1)
#     time.sleep(0.5)
#     little.move(1)
#     time.sleep(2)
#     open_hand()
#     
#     time.sleep(2)
#     thumb.move(1)
#     little.move(1)
#     ring.move(1)
#     time.sleep(2)
#     open_hand()

open_hand()
#close_hand(1)

try:
    while True:
        thumb.move(1)
        ring.move(1)
        little.move(1)
        
except KeyboardInterrupt:
    open_hand()