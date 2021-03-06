import logging
import math
import numpy as np
import random
import time
import cv2
import os

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
        self.dist = 0
        self.last_pos = [0,0]
        
        self._seed()
        self.stallCount = 0
        self.last_collision = None
        global airsimClient
        airsimClient = myAirSimCarClient()
        self.dirname = time.strftime("%Y_%m_%d_%H_%M") + '_rl' 
        os.mkdir(self.dirname)
        self.f = open(self.dirname+ '/log.txt','w')
        firstline = 'x,y,reward,collision\n'
        self.f.write(firstline)

         
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def computeReward(self, mode='roam'):
        speed = self.car_state.speed 
        steer = self.steer
        dSpeed = 0
        this_pos = [self.car_state.kinematics_estimated.position.x_val, self.car_state.kinematics_estimated.position.y_val]
        this_dist = ((self.last_pos[0]-this_pos[0])**2 + (self.last_pos[1]-this_pos[1])**2)** 0.5
        self.dist += this_dist
        self.last_pos = this_pos
        
        if mode == 'roam' or mode == 'smooth':
            
            reward = 0
            reward += this_dist
            
            '''
            # penalize sharp steering, to discourage going in a circle
            if abs(steer) >= 1.0 and speed > 100:
                reward -= abs(steer) * 2
            # penalize collision
            if len(self.allLogs['speed']) > 0:
                dSpeed = speed - self.allLogs['speed'][-2]
            else:
                dSpeed = 0
            '''
            
        if mode == 'smooth':
            # also penalize on jerky motion, based on a fake G-sensor
            steerLog = self.allLogs['steer']
            g = abs(steerLog[-1] - steerLog[-2]) * 5
            reward -= g
            
        return [reward, dSpeed]
    
        
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        time.sleep(0.05)
        car_state = airsimClient.getCarState()
        self.car_state = car_state
        speed = car_state.speed        
        
        self.stepN += 1
        steer = (action - 2)/2
        gas = max(min(15,(speed-15)/-12),0)
        
        airsimClient.setCarControls(gas, steer)                  
        self.steer = steer
        
        collision_info = airsimClient.simGetCollisionInfo()
        if collision_info.time_stamp != self.last_collision and collision_info.time_stamp != 0:
            done = True
        else:
            done = False
        
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
        
        # Training using the Roaming mode 
        reward, dSpeed = self.computeReward('roam')
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])

        # Terminate the episode on large cumulative amount penalties, 
        # since car probably got into an unexpected loop of some sort
        if rewardSum < -1000:
            done = True

        line = '%.2f,%.2f,%.2f,%s\n' % (car_state.kinematics_estimated.position.x_val, car_state.kinematics_estimated.position.y_val,rewardSum,collision_info.object_name)
        self.f.write(line)

        sys.stdout.write("\r\x1b[K{}/{}==>reward/depth/steer/speed: {:.0f}/{:.0f}   \t({:.1f}/{:.1f}/{:.1f})   \t{:.1f}/{:.1f}  \t{:.2f}/{:.2f}  ".format(self.episodeN, self.stepN, reward, rewardSum, self.state[0], self.state[1], self.state[2], steer, steerAverage, speed, dSpeed))
        sys.stdout.flush()
        
        
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
        self.dist = 0
        self.last_pos = [0,0]
        
        print("")
        
        self.allLogs = { 'speed': [0] }
        
        # Randomize the initial steering to broaden learning
        self.state = (100, 100, 100, random.uniform(0.0, 5.0))
        #self.state = (100, 100, 100, random.uniform(0.0, 21.0))
        self.f.close()
        self.f = open(self.dirname+ '/log.txt','a')
        return np.array(self.state)