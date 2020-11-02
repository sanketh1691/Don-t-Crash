import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#model_path = "models/first_model_more_data_1.h5"
#model_path = "models/first_model.h5"
model_path = "models/City.h5"
model = keras.models.load_model(model_path)
print('Loaded ML model')

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
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
    while 1:# get state of the car
        car_state = client.getCarState()
        print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
        
        car_controls.throttle = max(min(10,(car_state.speed-10)/-5),0)
        '''if car_state.speed < 5:
            car_controls.throttle = 1
        else:
            car_controls.throttle = 0

        # get camera images from the car
        
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
            airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
        print('Retrieved images: %d' % len(responses))
        '''
        
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene,False,False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        x_arr = img_rgb
        '''
        x_arr = rgb2gray(img_rgb)
        x_arr = x_arr.reshape(1,response.height, response.width,1)
        '''
        steering = model.predict(x_arr)
        print(steering)
        '''
        l_r = np.argmax(steering[0])
            
        if l_r == 0:
            car_controls.steering = -0.05
        elif l_r == 1:
            car_controls.steering = 0
        else:
            car_controls.steering = 0.05
        '''
        client.setCarControls(car_controls)
        
        print("Go Forward")
        #time.sleep(0.05)
        '''
        # go forward
        car_controls.throttle = 0.5
        car_controls.steering = 0
        client.setCarControls(car_controls)
        print("Go Forward")
        time.sleep(3)   # let car drive a bit

        # Go forward + steer right
        car_controls.throttle = 0.5
        car_controls.steering = 1
        client.setCarControls(car_controls)
        print("Go Forward, steer right")
        time.sleep(3)   # let car drive a bit

        # go reverse
        car_controls.throttle = -0.5
        car_controls.is_manual_gear = True
        car_controls.manual_gear = -1
        car_controls.steering = 0
        client.setCarControls(car_controls)
        print("Go reverse, steer right")
        time.sleep(3)   # let car drive a bit
        car_controls.is_manual_gear = False # change back gear to auto
        car_controls.manual_gear = 0

        # apply brakes
        car_controls.brake = 1
        client.setCarControls(car_controls)
        print("Apply brakes")
        time.sleep(3)   # let car drive a bit
        car_controls.brake = 0 #remove brake
        

        for response_idx, response in enumerate(responses):
            filename = os.path.join(tmp_dir, f"{0}_{response.image_type}_{response_idx}")

            if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress: #png format
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else: #uncompressed array
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
        '''
except:
    print('client broken')
    
    client.enableApiControl(False)
                
#restore to original state
#client.reset()

client.enableApiControl(False)
