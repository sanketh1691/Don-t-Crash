import logging
import math
import numpy as np
import random
import time
import cv2

import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box
import airsim

from envs.airsim.myAirSimCarClient import *

logger = logging.getLogger(__name__)

class AirSimCarEnv(gym.Env):

    airsimClient = None
    def __init__(self):
        # left depth, center depth, right depth, steering
        self.low = np.array([0.0, 0.0, 0.0, 0])
        self.high = np.array([100.0, 100.0, 100.0, 5])
        #self.high = np.array([100.0, 100.0, 100.0, 21])
        
        self.observation_space = spaces.Box(self.low, self.high)
        self.action_space = spaces.Discrete(5)
        #self.action_space = spaces.Discrete(21)
        
        self.state = (100, 100, 100, random.uniform(-1.0, 1.0))
        
        self.episodeN = 0
        self.stepN = 0 
        self.allLogs = { 'speed':[0] }
        
        self._seed()
        self.stallCount = 0
        self.last_collision = None
        global airsimClient
        airsimClient = myAirSimCarClient()
        
        self.LABELS = open("../AirSim/PythonClient/car/yolo/coco.names").read().strip().split("\n")
        self.path_weights = "../AirSim/PythonClient/car/yolo/yolov3.weights"
        self.path_config = "../AirSim/PythonClient/car/yolo/yolov3.cfg"        
        
        self.LABELS = open("../Yolo-Fastest/data/coco.names").read().strip().split("\n")
        self.path_weights = "../Yolo-Fastest/Yolo-Fastest/COCO/yolo-fastest.weights"
        self.path_config = "../Yolo-Fastest/Yolo-Fastest/COCO/yolo-fastest.cfg"
        
        np.random.seed(42)
        self.net = cv2.dnn.readNetFromDarknet(self.path_config, self.path_weights)
         
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def computeReward(self, mode='roam'):
        speed = self.car_state.speed 
        steer = self.steer
        dSpeed = 0
        
        '''
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            print("Collision at pos %s, normal %s, impact pt %s, penetration %f, name %s, obj id %d" % (
            pprint.pformat(collision_info.position), 
            pprint.pformat(collision_info.normal), 
            pprint.pformat(collision_info.impact_point), 
            collision_info.penetration_depth, collision_info.object_name, collision_info.object_id))
        break
        '''
        
        if mode == 'roam' or mode == 'smooth':
            # reward for speed
            reward = speed/60
            # penalize sharp steering, to discourage going in a circle
            if abs(steer) >= 1.0 and speed > 100:
                reward -= abs(steer) * 2
            # penalize collision
            if len(self.allLogs['speed']) > 0:
                dSpeed = speed - self.allLogs['speed'][-2]
            else:
                dSpeed = 0
            reward += dSpeed
            # penalize for going in a loop forever
            #reward -= abs(self.steerAverage) * 10
        else:
            reward = 1
            # Placehoder. To be filled

        if mode == 'smooth':
            # also penalize on jerky motion, based on a fake G-sensor
            steerLog = self.allLogs['steer']
            g = abs(steerLog[-1] - steerLog[-2]) * 5
            reward -= g
            
        return [reward, dSpeed]
    

    def get_closeness(self,image,x1,y1,w1,h1):
        #color2 = [110,220,69]
        k = 0.1
        start_pt = (int(x1+w1/2),144)
        end_pt = (int(x1+w1/2),y1+h1)
        #cv2.line(image, start_pt, end_pt, color2,2)
        distance = (start_pt[0]-end_pt[0])**2 + (start_pt[1]-end_pt[1])**2
        distance = math.sqrt(distance)
        area = w1*h1
        return round(k*area/distance,3)

    def yolo(self, img, net, confidence_threshold, threshold):
        #print("In YOLO method")
        #image = cv2.imread(img)
        image = img
        
        #print("shape of image is:",image.shape)
        #since image is 4 channel, we consider height and width
        (H, W) = image.shape[:2]
        layerNames = net.getLayerNames()
        layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        stt = time.time()
        
        ##construct a blob from the input image
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0,(416,416),swapRB=True,crop=False)
        net.setInput(blob)
        netOutputs = net.forward(layerNames)
        
        
        print('1 %.3f' % (time.time() - stt))
        # Constructing bounding box
        boxes = []
        confidences = []
        classIDs = []

        for output in netOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if(confidence>confidence_threshold):
                    box = detection[0:4] * np.array([W, H, W, H])
                    (Xcenter, Ycenter, width, height) = box.astype("int")
                    #obtain coordinates for top left corner
                    tl_x = int(Xcenter - (width/2))
                    tl_y = int(Ycenter - (height/2))
                    boxes.append([tl_x,tl_y,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        
        print('2 %.3f' % (time.time() - stt))
        
        
        #applying non maxima suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,threshold)
        close = dict()
        if(len(idxs)>0):
            for i in idxs.flatten():
                (x,y) = (boxes[i][0], boxes[i][1])
                (w,h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x,y), (x+w,y+h), (100,220,210), 2)
                text = self.LABELS[classIDs[i]]
                cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, (180,100,70), 1)
                cl = self.get_closeness(image,x,y,w,h)
                close.update({(self.LABELS[classIDs[i]],confidences[i]):[cl,(x+w)/2]})
        #df = pd.DataFrame(columns=['Object','Confidence','Closeness'])
        close_metric = sorted(close.items(), key=lambda x: x[1], reverse=True)
        print('4 %.3f' % (time.time() - stt))
        
        
        return image, close_metric

        
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        time.sleep(0.05)
        car_state = airsimClient.getCarState()
        self.car_state = car_state
        speed = car_state.speed        
        
        self.stepN += 1
        steer = (action - 2)
        gas = max(min(20,(speed-20)/-15),0)
        
        airsimClient.setCarControls(gas, steer)                  
        self.steer = steer
        
        collision_info = airsimClient.simGetCollisionInfo()
        if collision_info.time_stamp != self.last_collision and collision_info.time_stamp != 0:
            done = True
        else:
            done = False
        
        '''
        elif speed < 0.5:
            self.stallCount += 1
        else:
            self.stallCount = 0
        if self.stallCount > 6:
            done = False
        else:
            done = False
        '''
        
        self.last_collision = collision_info.time_stamp
        
        self.sensors = airsimClient.getSensorStates()
        cdepth = self.sensors[1]
        self.state = self.sensors
        self.state.append(action)

        self.addToLog('speed', speed)
        self.addToLog('steer', steer)
        steerLookback = 17
        steerAverage = np.average(self.allLogs['steer'][-steerLookback:])   
        self.steerAverage = steerAverage
        
        responses2 = airsimClient.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        cam_image = responses2[0]
        img1d = np.fromstring(cam_image.image_data_uint8, dtype=np.uint8) 
        img_rgb = img1d.reshape(cam_image.height, cam_image.width, 3)

        yolores, closeness = self.yolo(img_rgb,self.net,0.5,0.5)
        print()
        print(closeness)
        
        # Training using the Roaming mode 
        reward, dSpeed = self.computeReward('roam')
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])

        # Terminate the episode on large cumulative amount penalties, 
        # since car probably got into an unexpected loop of some sort
        if rewardSum < -1000:
            done = True
            
        sys.stdout.write("\r\x1b[K{}/{}==>reward/depth/steer/speed: {:.0f}/{:.0f}   \t({:.1f}/{:.1f}/{:.1f})   \t{:.1f}/{:.1f}  \t{:.2f}/{:.2f}  ".format(self.episodeN, self.stepN, reward, rewardSum, self.state[0], self.state[1], self.state[2], steer, steerAverage, speed, dSpeed))
        sys.stdout.flush()
        
        # placeholder for additional logic
        if done:
            pass

        return np.array(self.state), reward, done, {}

    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def _reset(self):
        airsimClient.reset()
        airsimClient.setCarControls(1, 0)
        time.sleep(0.8)
        
        self.stepN = 0
        self.stallCount = 0
        self.episodeN += 1
        
        print("")
        
        self.allLogs = { 'speed': [0] }
        
        # Randomize the initial steering to broaden learning
        self.state = (100, 100, 100, random.uniform(0.0, 5.0))
        #self.state = (100, 100, 100, random.uniform(0.0, 21.0))
        return np.array(self.state)