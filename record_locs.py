import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile

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

try:
#    while 1:# get state of the car
    for hi in range(1):
        car_state = client.getCarState()
        print(car_state.kinematics_estimated.position)
        
except:
    print('client broken')
             

client.enableApiControl(False)
