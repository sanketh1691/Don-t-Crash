import setup_path 
import airsim

import os
import traceback
import math
import time
from argparse import ArgumentParser
from numpy import ones,vstack
from numpy.linalg import lstsq


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
        exp = np.random.rand() - 0.1 < 0.25
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
                Dense(3, init=he_uniform(scale=0.01)),
                Dense(4, init=he_uniform(scale=0.01)),
                #Dense(4, init=he_uniform(scale=0.01)),
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
        vm_schedule = momentum_schedule(0.3)
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
        #print(self._num_actions_taken)
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
            if self._num_trains % 100 == 0:
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

def interpret_action(action):
    if action == 0:
        car_controls.brake = 0
        car_controls.throttle = 1
        car_controls.steering = 0
        '''
    elif action == 1:
        car_controls.brake = 0
        car_controls.throttle = 1
        car_controls.steering = 0
        '''
    elif action == 1:
        car_controls.brake = 0
        car_controls.throttle = 1
        car_controls.steering = 0.5
    elif action == 2:
        car_controls.brake = 0
        car_controls.throttle = 1
        car_controls.steering = -0.5
    elif action == 3:
        car_controls.brake = 0
        car_controls.throttle = 0
        car_controls.steering = 0.5
    else:
        car_controls.brake = 0
        car_controls.throttle = 0
        car_controls.steering = -0.5

    return car_controls

def line_equation(xs, ys):
    points = [(xs[0],ys[0]),(xs[1],ys[1])]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    return m , c
    
def distance_from(point,coef):
    num = ((coef[0]*point[0])-point[1]+coef[1])/np.sqrt((coef[0]*coef[0])+1)
    return num
    
def dist_from(point, magnitude=True):
    dist = 100000000
    linenum = -1
    val = 0
    for i in range(len(pts)):
        value = distance_from(point,coeffs[i])
        this_dist = max(-value,value)
        if this_dist < dist:
            dist = this_dist
            val = value
            linenum = i
    if magnitude:
        return max(dist, -dist), linenum
    return val

def compute_reward(car_state,distance, angle, record=False):
    MAX_SPEED = 25
    MIN_SPEED = 0.05
    thresh_dist = 15
    beta = 3

    z = 0
    pd = car_state.kinematics_estimated.position
    car_pt = np.array([pd.x_val, pd.y_val])
    
    dist, linenum = dist_from(car_pt, magnitude=True)
    '''
    dist = 10000000
    linenum = -1
    for i in range(len(pts)):
        this_dist = dist_from(car_pt,magnitude=True)
        #this_dist = np.linalg.norm(np.cross(np.array(pts[i][1])-np.array(pts[i][0]), np.array(pts[i][0])-np.array(car_pt)))/np.linalg.norm(
         #           np.array(pts[i][1])-np.array(pts[i][0]))
        if this_dist < dist:
            dist = this_dist
            linenum = i
    '''      
    if dist > thresh_dist:
        reward = -50
        if record:
            line_w = '0,0,0,0,0,0,0\n'
    else:
        reward_dist =  15-(dist*2)
        reward_angle = 0.5 * (20-max(angle,-angle)/5)
        try:
            reward_speed = 6 * math.log(car_state.speed-0.5) + 1
        except:
            reward_speed = 0
        reward = reward_dist + reward_angle + reward_speed# + distance/10
        #reward = 10 # + distance/10
        if record:
            line_w = '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (car_pt[0],car_pt[1],car_state.speed,reward_speed,reward_dist,reward_angle,reward)
            print('Dist %.2f, Angle %.2f, Speed %.2f, Distance_tot %.2f' %(reward_dist, reward_angle, reward_speed, distance/10))
        print('\nReward: %.3f\n'%reward)
    
    f.write(line_w)
    return reward, linenum

def isDone(car_state, car_controls, reward):
    done = 0
    if reward < -1:
        done = 1
    if car_controls.brake == 0:
        if car_state.speed <= 0.5:
            done = 1
    return done

def compute_angle(car_state, last_pos, line):
    cs = car_state.kinematics_estimated.position
    pos = [cs.x_val, cs.y_val]
    try:
        #line_angle = math.atan(-(pts[line][0]-pts[line][0])/(pts[line][1]-pts[line][1]))*180/math.pi
        line_angle = math.atan(-(pts[line][1]-pts[line][1])/(pts[line][0]-pts[line][0]))*180/math.pi
        current_angle = math.atan((pos[1]-last_pos[1])/(pos[0]-last_pos[0]))* 180/math.pi
    except:
        line_angle = 0
        current_angle = 0
    print('line_angle: %.1f current_angle: %.1f' %(line_angle, current_angle))
    if max(line_angle-current_angle,current_angle-line_angle) < 90:
        return line_angle - current_angle
    else:
        if -90 < 180 - (line_angle-current_angle) < 90:
            return -(180 - (line_angle-current_angle))
        return -(180 + line_angle-current_angle)
   
client = airsim.CarClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
#client.enableApiControl(False)
car_controls = airsim.CarControls()

# Make RL agent
NumBufferFrames = 4
SizeState = 3 #speed, angle from circle tangent, distance from circumference (+ve-ve)
NumActions = 5
agent = DeepQAgent((NumBufferFrames, SizeState), NumActions, monitor=True)
agent._trainer.restore_from_checkpoint('Good_dqn_run/model49525')


# Train
epoch = 100
current_step = 0
max_steps = epoch * 250000

zero_controls = car_controls

current_state = np.zeros(SizeState)
distance = 0
angle = 0
cs = client.getCarState()
last_pos = [cs.kinematics_estimated.position.x_val,cs.kinematics_estimated.position.y_val]
record = True

pts = [[[-39.04078, -810.64233],[-22.38516,-21.23539]],
      [[-597.26538, -596.10773],[-292.65579, 248.84686]],
      [[25.66422, 25.34047],[-81.8964, -916.9816]],
      [[0, -39.0408], [0, -22.38516]],
      [[-39.0408, 25.6642],[-22.3852, -81.89641]]]
 
coeffs = [line_equation(i[0], i[1]) for i in pts]

#rev_pts = [j[::-1] for i in pts for j in i]
#pts += [[rev_pts[i], rev_pts[i+1]]for i in range(0,len(pts),2)]

car_controls.steering = 10
car_controls.throttle = 1
client.setCarControls(car_controls)
sleep_time = 0.95
time.sleep(sleep_time)   

try:
    # record log of positions and reward
    if record:
        dirname = time.strftime("%Y_%m_%d_%H_%M") + '_dqn_city' 
        os.mkdir(dirname)
        f = open(dirname+ '/log.txt','w')
        firstline = 'x,y,speed,reward_speed,reward_dist,reward\n'
        
    while True:
        action = agent.act(current_state)
        print(action)
        car_controls = interpret_action(action)
        client.setCarControls(car_controls)
        
        time.sleep(0.06)
        car_state = client.getCarState()
        reward, line = compute_reward(car_state,distance, angle,record)
        
        done = isDone(car_state, car_controls, reward)
        if done == 1:
            reward = -50

        agent.observe(current_state, action, reward, done)
        
        if done:
            #agent.train()
            client.reset()
            
            car_controls.steering = 10
            car_controls.throttle = 1
            client.setCarControls(car_controls)
            time.sleep(sleep_time)   
            #car_control = interpret_action(0)
            #client.setCarControls(car_control)
            print('Sleep then GO!\n')
            #time.sleep(0.5)
            current_step += 1
            cs = client.getCarState()
            last_pos = [cs.kinematics_estimated.position.x_val,cs.kinematics_estimated.position.y_val]
            distance = 0
        
        cs = client.getCarState()
        angle = compute_angle(cs, last_pos, line)
        this_pos = [cs.kinematics_estimated.position.x_val, cs.kinematics_estimated.position.y_val]
        input_dist =  dist_from(this_pos,magnitude=False)
        print('Angle: %.2f, Distance: %.2f' %(angle, input_dist))

        distance += ((last_pos[0]-this_pos[0])**2 + (last_pos[1]-this_pos[1])**2)** 0.5
        last_pos = this_pos
        current_state = np.array([cs.speed, angle, input_dist])
        
except KeyboardInterrupt as inst:
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
