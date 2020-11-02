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
        print(client.simListSceneObjects())
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene), #0
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanner), #1 
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective), #2
            airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #3
            airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized), #4
            airsim.ImageRequest("0", airsim.ImageType.Segmentation), #5
            airsim.ImageRequest("0", airsim.ImageType.SurfaceNormals), #6
            airsim.ImageRequest("0", airsim.ImageType.Infrared)]) #7
        print('Retrieved images: %d' % len(responses))
        
        
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
        
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene,False,False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
       


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
