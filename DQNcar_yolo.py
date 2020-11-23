import setup_path 
import airsim

import math
import time
from argparse import ArgumentParser
import os
import cv2
import sys


#import gym #pip install gym
import numpy as np
from cntk.core import Value #pip install cntk-gpu
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer

import pickle

class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.

        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.

        Attributes:
            size (int): The minibatch size

        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.

        Attributes:
            size (int): Minibatch size

        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.

        Attributes:
            index (int): State's index

        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis

        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history

        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0

        """
        self._buffer.fill(0)

class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)

        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start
        """
        epsilon_step =  (1-0.001)**step
        print('Epsilon: %.3f,  Step: %d' %(epsilon_step, step))
        return epsilon_step
        
    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        exp = np.random.rand() - 0.1 < self._epsilon(step)
        #exp = np.random.rand() - 0.1 < 0.25
        print('Exploring?' ,exp)
        return exp

def huber_loss(y, y_hat, delta):
    """ Compute the Huber Loss as part of the model graph

    Huber Loss is more robust to outliers. It is defined as:
     if |y - y_hat| < delta :
        0.5 * (y - y_hat)**2
    else :
        delta * |y - y_hat| - 0.5 * delta**2

    Attributes:
        y (Tensor[-1, 1]): Target value
        y_hat(Tensor[-1, 1]): Estimated value
        delta (float): Outliers threshold

    Returns:
        CNTK Graph Node
    """
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * square(error)
    more_than = (delta * abs_error) - half_delta_squared
    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

    return reduce_sum(loss_per_sample, name='loss')

class DeepQAgent(object):
    """
    Implementation of Deep Q Neural Network agent like in:
        Nature 518. "Human-level control through deep reinforcement learning" (Mnih & al. 2015)
    """
    def __init__(self, input_shape, nb_actions,
                 gamma=0.7, explorer=LinearEpsilonAnnealingExplorer(1, 0.1, 100000),
                 learning_rate=0.001, momentum=0.2, minibatch_size=32,
                 memory_size=500000, train_after=100, train_interval=100, target_update_interval=100,
                 monitor=True):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0
        self._num_trains = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        with default_options(activation=relu, init=he_uniform()):
            self._action_value_net = Sequential([
                Dense(5, init=he_uniform(scale=0.01)),
                Dense(8, init=he_uniform(scale=0.01)),
                #Dense(16, init=he_uniform(scale=0.01)),
                #Dense(32, init=he_uniform(scale=0.01)),
                Dense(nb_actions, activation=None, init=he_uniform(scale=0.01))
            ])
        
        self._action_value_net.update_signature(Tensor[input_shape])

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self._target_net = self._action_value_net.clone(CloneMethod.freeze)

        # Function computing Q-values targets as part of the computation graph
        @Function
        @Signature(post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def compute_q_targets(post_states, rewards, terminals):
            return element_select(
                terminals,
                rewards,
                gamma * reduce_max(self._target_net(post_states), axis=0) + rewards,
            )

        # Define the loss, using Huber Loss (more robust to outliers)
        @Function
        @Signature(pre_states=Tensor[input_shape], actions=Tensor[nb_actions],
                   post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def criterion(pre_states, actions, post_states, rewards, terminals):
            # Compute the q_targets
            q_targets = compute_q_targets(post_states, rewards, terminals)

            # actions is a 1-hot encoding of the action done by the agent
            q_acted = reduce_sum(self._action_value_net(pre_states) * actions, axis=0)

            # Define training criterion as the Huber Loss function
            return huber_loss(q_targets, q_acted, 1.0)

        # Adam based SGD
        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._action_value_net.parameters, lr_schedule,
                     momentum=m_schedule, variance_momentum=vm_schedule)

        self._metrics_writer = TensorBoardProgressWriter(freq=1, log_dir='metrics', model=criterion) if monitor else None
        self._learner = l_sgd
        self._trainer = Trainer(criterion, (criterion, None), l_sgd, self._metrics_writer)

        #self._trainer.restore_from_checkpoint('models/oldmodels/model800000')

    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.

        Attributes:
            state (Tensor[input_shape]): The current environment state

        Returns: Int >= 0 : Next action to do
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        # If policy requires agent to explore, sample random action
        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.nb_actions)
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            q_values = self._action_value_net.eval(
                # Append batch axis with only one sample to evaluate
                env_with_history.reshape((1,) + env_with_history.shape)
            )

            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))

            # Return the value maximizing the expected reward
            action = q_values.argmax()

        # Keep track of interval action counter
        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state

        Attributes:
            old_state (Tensor[input_shape]): Previous environment state
            action (int): Action done by the agent
            reward (float): Reward for doing this action in the old_state environment
            done (bool): Indicate if the action has terminated the environment
        """
        self._episode_rewards.append(reward)

        # If done, reset short term memory (ie. History)
        if done:
            # Plot the metrics through Tensorboard and reset buffers
            if self._metrics_writer is not None:
                self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self):
        """ This allows the agent to train itself to better understand the environment dynamics.
        The agent will compute the expected reward for the state(t+1)
        and update the expected reward at step t according to this.

        The target expectation is computed through the Target Network, which is a more stable version
        of the Action Value Network for increasing training stability.

        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.
        """

        agent_step = self._num_actions_taken

        if agent_step >= self._train_after:
            #if (agent_step % self._train_interval) == 0:
            self._num_trains += 1
            if self._num_trains % 3 == 0:
                print('\nTraining minibatch\n')
                client.setCarControls(zero_controls)
                pre_states, actions, post_states, rewards, terminals = self._memory.minibatch(self._minibatch_size)
                self._trainer.train_minibatch(
                    self._trainer.loss_function.argument_map(
                        pre_states=pre_states,
                        actions=Value.one_hot(actions.reshape(-1, 1).tolist(), self.nb_actions),
                        post_states=post_states,
                        rewards=rewards,
                        terminals=terminals
                    )
                )
            # Update the Target Network if needed
            if self._num_trains % 30 == 0:
                print('updating network')
                self._target_net = self._action_value_net.clone(CloneMethod.freeze)
                filename = dirname+"\model%d" % agent_step
                self._trainer.save_checkpoint(filename)
                
    def _plot_metrics(self):
        """Plot current buffers accumulated values to visualize agent learning
        """
        if len(self._episode_q_means) > 0:
            mean_q = np.asscalar(np.mean(self._episode_q_means))
            self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._num_actions_taken)

        if len(self._episode_q_stddev) > 0:
            std_q = np.asscalar(np.mean(self._episode_q_stddev))
            self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._num_actions_taken)

        self._metrics_writer.write_value('Sum rewards per ep.', sum(self._episode_rewards), self._num_actions_taken)


def get_closeness(image,x1,y1,w1,h1):
    color2 = [110,220,69]
    k = 3
    start_pt = (int(x1+w1/2),144)
    end_pt = (int(x1+w1/2),y1+h1)
    #cv2.line(image, start_pt, end_pt, color2,2)
    distance = (start_pt[0]-end_pt[0])**2 + (start_pt[1]-end_pt[1])**2
    distance = math.sqrt(distance)
    area = w1*h1
    return round(k*area/(distance+0.0000000001),3)

def yolo(img, net, confidence_threshold, threshold):
    #image = cv2.imread(img)
    image  = img
    #since image is 4 channel, we consider height and width
    (H, W) = image.shape[:2]
    layerNames = net.getLayerNames()
    layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ##construct a blob from the input image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    netOutputs = net.forward(layerNames)
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
                
    #applying non maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,
                            threshold)
    close = dict()
    if(len(idxs)>0):
        for i in idxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(image, (x,y), (x+w,y+h), (100,220,210), 2)
            text = LABELS[classIDs[i]]
            cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (180,100,70), 1)
            cl = get_closeness(image,x,y,w,h)
            close.update({(LABELS[classIDs[i]],confidences[i]):[cl,x+w/2,int((x+w/2)>128)]})
    close_metric = sorted(close.items(), key=lambda x: x[1], reverse=True)
    close_l = 0
    close_r = 0
    for i in close_metric:
        if i[1][-1]:
            close_r = max(i[1][0],close_r)
        else:
            close_l = max(i[1][0],close_l)
            
    return image, close_l, close_r
    
def interpret_action(action, car_state):
    car_controls.throttle = max(min(20,(car_state.speed-20)/-15),0)
    if action == 0:
        car_controls.steering = 0
    elif action == 1:
        car_controls.steering = 1.5
    elif action == 2:
        car_controls.steering = -1.5
    elif action == 3:
        car_controls.steering = 0.75
    else:
        car_controls.steering = -0.75
        
    return car_controls


def compute_reward(car_state,last_pos, distance,last_collision):

    z = 0
    pd = car_state.kinematics_estimated.position
    car_pt = np.array([pd.x_val, pd.y_val])    
    this_dist = ((last_pos[0]-car_pt[0])**2 + (last_pos[1]-car_pt[1])**2)** 0.5
    distance += this_dist
    last_pos = [car_pt[0], car_pt[1]]

    collision_info = client.simGetCollisionInfo()
    if collision_info.time_stamp != last_collision and collision_info.time_stamp != 0:
        reward = -50
        line = '0,0,0,0\n'
    else:
        reward = 0
        reward -= (close_r + close_l) * 0.3
        reward = this_dist
        line = '%.2f,%.2f,%.2f,%.2f\n' % (car_pt[0], car_pt[1], this_dist, reward)
        print('Dist %.2f, Distance_tot %.2f' %(this_dist, distance))
        print('\nReward: %.3f\n'%reward)
        
    last_collision = collision_info.time_stamp
    
    return reward, last_pos, distance, last_collision

def isDone(car_state, car_controls, reward):
    done = 0
    if reward < -1:
        done = 1
    if car_controls.brake == 0:
        if car_state.speed <= 0.5:
            done = 1
    return done
    
    
def getScore(imgFull, hCenter, wCenter, size, checkMin):
    wsize2 = size/2
    
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

def getSensorStates2(img, h, w, size):
    h2 = h/2
    w2 = w/2 
    offset = 50
    cscore = getScore(img, h2, w2, size, True)
    lscore = getScore(img, h2, w2-offset, size, True)
    rscore = getScore(img, h2, w2+offset, size, True)
    return [lscore, cscore, rscore]

def getSensorStates(img1,img2):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    response = responses[0]
    img1 = img2
    try:
        img2 = np.array(response.image_data_float).reshape(144,256)
    except:
        img2 = None
    img2_ = img2
    result = [100.0, 100.0, 100.0]
    
    if img2_ is not None:    
        if len(img2_) > 1:
            h = 144
            w = 256
            size = 20
            if img1 is not None and img2_ is not None:
                result = getSensorStates2(img2_, h, w, size)
    
    return result, img1, img2
    
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
car_state = client.getCarState()
        
zero_controls = car_controls
# Make RL agent
NumBufferFrames = 4
SizeState = 5
NumActions = 5
agent = DeepQAgent((NumBufferFrames, SizeState), NumActions, monitor=True)
current_state = np.zeros(SizeState)

#Loading YOLO
print("loading YOLO")
LABELS = open("../../../Yolo-Fastest/data/coco.names").read().strip().split("\n")
path_weights = "../../../Yolo-Fastest/Yolo-Fastest/COCO/yolo-fastest.weights"
path_config = "../../../Yolo-Fastest/Yolo-Fastest/COCO/yolo-fastest.cfg"
np.random.seed(42)
#load the trained YOLO net using dnn library in cv2
net = cv2.dnn.readNetFromDarknet(path_config, path_weights)

# Train
epoch = 100
current_step = 0
max_steps = epoch * 250000

responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
cam_image = responses[0]
img1d = np.fromstring(cam_image.image_data_uint8, dtype=np.uint8) 
img_rgb = img1d.reshape(cam_image.height, cam_image.width, 3)
yolores, close_r, close_l = yolo(img_rgb,net,0.5,0.5)
img_idx=0
last_collision = 0
sensors, img1, img2 = getSensorStates(None,None)
state = sensors
distance = 0
last_pos = [0,0]

try:
    dirname = time.strftime("%Y_%m_%d_%H_%M") + '_yolo'
    os.mkdir(dirname)
    f = open(dirname+ '/log.txt','w')
    firstline = 'x,y,distance,reward\n'

    while True:

        action = agent.act(current_state)
        car_controls = interpret_action(action, car_state)
        client.setCarControls(car_controls)

        time.sleep(0.1)
        sensors, img1, img2 = getSensorStates(img1,img2)
        car_state = client.getCarState()
        #cdepth = sensors[1]
        state = sensors
        
        reward, last_pos, distance, last_collision= compute_reward(car_state, last_pos, distance, last_collision)        
        done = isDone(car_state, car_controls, reward)
        if done == 1:
            reward = -10

        agent.observe(current_state, action, reward, done)
        
        if done:
            agent.train()
            client.reset()
            car_controls.throttle = 1
            car_controls.steering = 0
            client.setCarControls(car_controls)
            print('Sleep then GO!\n')
            time.sleep(2.5)
            current_step += 1
            cs = client.getCarState()
            sensors, img1, img2 = getSensorStates(None,None)
            #cdepth = sensors[1]
            state = sensors
            #state.append(action)
            last_pos = [cs.kinematics_estimated.position.x_val,cs.kinematics_estimated.position.y_val]
            distance = 0

        #print("applying YOLO")
        #filepath = "yolo/image_outputs/"
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        cam_image = responses[0]
        # get numpy array
        img1d = np.fromstring(cam_image.image_data_uint8, dtype=np.uint8) 

        # reshape array to 4 channel image array H X W X 4
        try:       
            img_rgb = img1d.reshape(cam_image.height, cam_image.width, 3)
        except:
            continue
        # original image is fliped vertically
        #img_rgb = np.flipud(img_rgb)
        #filename = str(img_idx)+'.png'
        #airsim.write_png(os.path.normpath(filepath + filename), img_rgb) 
        #cam_image = client.simGetImage("0", airsim.ImageType.Scene)

        #image_name = os.path.normpath(filepath + filename)
        #yolores = yolo(cam_image.image_data_uint8,net,0.5,0.5)
        #yolores, closeness = yolo(image_name,net,0.5,0.5)
        yolores, close_r, close_l = yolo(img_rgb,net,0.5,0.5)
        #yolores.save(filepath+str(img_idx)+"yolo.png")
        #cv2.imwrite(filepath+str(img_idx)+"yolo.png",yolores)
        #print("saved image")
        img_idx+=1
        current_state = np.array(state + [close_r, close_l])
        print(current_state)
            
        
except KeyboardInterrupt as inst:
    import traceback
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    print(inst)
    traceback.print_exc() 
    save = input('save model and log (y/n): ')
    
    if save.lower() == 'n':
        import shutil    
        f.close()
        shutil.rmtree(dirname)
        
client.enableApiControl(False)