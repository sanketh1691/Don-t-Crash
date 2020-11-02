import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile
import pandas as pd
#from datetime import datetime
import time

'''
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
'''

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
print ("Saving images to %s" % tmp_dir)

try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise
        
dirname = time.strftime("%Y_%m_%d_%H_%M")
os.mkdir(dirname)

f = open(dirname+ '/log.txt','w')

try:
    while 1:
        car_state = client.getCarState()
        pos = car_state.kinematics_estimated.position
        line = str(pos.x_val)+','+str(pos.y_val)+','+str(pos.z_val)+'\n'
        f.write(line)
        time.sleep(0.1)

            
        
except:
    print('client broken')
             
f.close()
client.enableApiControl(False)
