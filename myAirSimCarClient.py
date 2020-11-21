import numpy as np
import sys, os

# Change the path below to point to the directoy where you installed the AirSim PythonClient
sys.path.append('..\..\..\AirSim\PythonClient')
#from AirSimClient import *
from airsim import * 

class myAirSimCarClient(CarClient):

    def __init__(self):        
        self.img1 = None
        self.img2 = None

        CarClient.__init__(self)
        CarClient.confirmConnection(self)
        self.enableApiControl(True)
        
    def getScore(self, imgFull, hCenter, wCenter, size, checkMin):
        wsize2 = size /2
        
        hRange = range(int(hCenter-wsize2), int(hCenter+wsize2))
        wRange = range(int(wCenter-wsize2), int(wCenter+wsize2))
        
        sum = 0
        winMin = truemin = 9999999
        
        for i in hRange:
            for j in wRange:
                dist = imgFull[i,j]
                if checkMin:
                    winMin = min(dist, winMin)
                else:
                    winMin = dist
                truemin = min(truemin,dist)
                sum += winMin
        result =sum/size/size
       
        return result
       
    def setCarControls(self, gas, steer):
        car_controls = CarControls()
        car_controls.throttle = float(gas)
        car_controls.steering = float(steer)
        car_controls.is_manual_gear = False
        CarClient.setCarControls(self, car_controls)

          
    def getSensorStates2(self, img, h, w, size):
        h2 = h/2
        w2 = w/2 
        offset = 50
        cscore = self.getScore(img, h2, w2, size, True)
        lscore = self.getScore(img, h2, w2-offset, size, True)
        rscore = self.getScore(img, h2, w2+offset, size, True)
        return [lscore, cscore, rscore]

    def getSensorStates(self):
        responses = CarClient.simGetImages(self, [ImageRequest("0", ImageType.DepthPerspective, True, False)])
        #responses = CarClient.simGetImages(self, [ImageRequest("0", ImageType.Scene, True, False)])
        response = responses[0]
        self.img1 = self.img2
        try:
            self.img2 = np.array(response.image_data_float).reshape(144,256)
        except:
            self.img2 = None
        img2 = self.img2
        result = [100.0, 100.0, 100.0]
        
        if img2 is not None:    
            if len(img2) > 1:
                h = 144
                w = 256
                size = 20
                if self.img1 is not None and img2 is not None:
                    result = self.getSensorStates2(img2, h, w, size)
        return result
        
    def breakClient(self):
        self.enableApiControl(False)
        return
    

