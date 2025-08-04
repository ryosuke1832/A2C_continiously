import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time

# For LSL
import random
import time
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream  # , resolve_stream_byprop


class SoccerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self): #, streams_robot, streams_PEN, stream_sigma):

        # Init LSL streams

        stream_sigma= StreamInfo('MarkerStream', 'Sigma', 2, 0, 'float32', 'xxx')
        print("Create Stream 2")
        streams_robot = resolve_stream('name', 'DataStream')
        # streams_robot = resolve_stream('name', 'EEGLAB')
        print("Create Stream 1")
        streams_PEN = resolve_stream('name', 'MyMarkerStream') # resolve_stream('type', 'Markers')
        print("Create Stream 3")

        # create a new inlet to read from the stream
        self.inlet_robot = StreamInlet(streams_robot[0])

        # create a new outlet to send data to the stream
        self.outlet = StreamOutlet(stream_sigma)

        # create a new inlet to read from the stream
        self.inlet_PEN = StreamInlet(streams_PEN[0])

        # Actions Space
        # self.action_space = spaces.Box(
        #     low=np.array([0.25, 0.35]), high=np.array([0.45, 0.45]), dtype=np.float16)
        self.action_space = spaces.Box(
            low=0.25, high=0.45, shape=(1, 1), dtype=np.float16)

        # The observation Space
        self.observation_space = self._get_observation_space()

        # spaces.Box(
        # low=np.array([0.15, 0.25]), high=np.array([0.35, 0.45]), dtype=np.float16)

        # self.SET_TIME = self.gameTime  # 20-minutes
        # self.start_time = self.currentTime

        # Set Initial state of gym game environment

    def step(self, action):

        # Take an action -must return next state,
        # reward for current state,
        # is done or not
        # Info (could be empty
        # if not self.action_space.contains(action):
        #     if action[0]>self.action_space.high[0]:
        #         diff = action[0] - self.action_space.high[0]
        #         if diff > self.action_space.high[0]:
        #             action1 = self.action_space.high[0]
        #         else:
        #             action1 = action[0] - self.action_space.high[0] + self.action_space.low[0]
        #     elif action[0]< self.action_space.low[0]:
        #         action1=self.action_space.low[0]
        #     else:
        #         action1=action[0]
        #
        #     if action[0]>self.action_space.high[1]:
        #         diff=action[0]-self.action_space.high[1]
        #         if diff > self.action_space.high[1]:
        #             action2=self.action_space.high[1]
        #         else:
        #             action2 = action[0] - self.action_space.high[1] + self.action_space.low[1]
        #     elif action[0]<self.action_space.low[1]:
        #         action2=self.action_space.low[1]
        #     else:
        #         action2=action[0]
        #
        #     self.outlet.push_sample([action1, action2])
        # else:
        #     action1=action[0]
        #     action2=action[0]
        #     self.outlet.push_sample([action1, action2+0.1])

        if not self.action_space.contains(action):
            if action[0]>self.action_space.high:
                diff = action[0] - self.action_space.high
                if diff > self.action_space.high:
                    action1 = self.action_space.high
                else:
                    action1 = action[0] - self.action_space.high + self.action_space.low
            elif action[0]< self.action_space.low:
                action1=self.action_space.low
            else:
                action1=action[0]

            if action[0]>self.action_space.high:
                diff=action[0]-self.action_space.high
                if diff > self.action_space.high:
                    action2=self.action_space.high
                else:
                    action2 = action[0] - self.action_space.high + self.action_space.low
            elif action[0]<self.action_space.low:
                action2=self.action_space.low
            else:
                action2=action[0]

            self.outlet.push_sample([action1, action2])
        else:
            action1=action[0]
            action2=action[0]
            self.outlet.push_sample([action1, action2+0.1])


        # action1=action[0]
        # action2=action[0]
        # self.outlet.push_sample([0.25, action[0]])
        print('Action '+str([action1, action2]))

        self.state = self._get_current_state()

        # Check if the environment state is contained in the observation space
        #if not self.observation_space.contains(self.state):
        #    print('InvalidState')
            # raise InvalidStateError()

        # Assign reward
        rewards = self.inlet_PEN.pull_sample()[0]
        print('Rewards '+ str(rewards))
        reward=0
        if rewards[0] == 0:
            reward = 100
        elif rewards[0] ==1:
            reward = 50
        else:
            reward = -100

        print('Reward ' + str(reward))

        # If total time of session is completed
        if (time.time() - self.currentTime)/60 >=self.gameTime:
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def _setTimes(self,currentTime,gameTime):
        self.currentTime=currentTime
        self.gameTime=gameTime

        return 0

    def reset(self):

        self.outlet.push_sample([0.15, 0.25])
        # self.outlet.push_sample("0.25")
        self.state = self._get_current_state()

        return self.state

    # reset the game

    def _get_observation_space(self):
        """Get environment observation space.
        Returns:
            gym.spaces: Gym observation space object.
        """

        """
        Data as per Robot:
  
        const char * channels[] = {"sigma1", "sigma2", "sigma3", "sigma4", "sigma5", "sigma6", "int_force_x", "int_force_y",
                     "int_force_z", "int_force_xrot", "int_force_yrot", "int_force_zrot", "tcp_vel_x",
                     "tcp_vel_y", "tcp_vel_z", "tcp_vel_xrot", "tcp_vel_yrot", "tcp_vel_zrot", "tcp_pos_x",
                     "tcp_pos_y", "tcp_pos_z", "tcp_pos_xrot", "tcp_pos_yrot", "tcp_pos_zrot", "time", "sing1",
                     "sing2", "sing3", "sing4", "sing5", "sing6", "mode", "towordsSing1", "towordsSing2"};
  
        They are: whether or not sing avoidance is on (3 linear and 3 angular), interaction forces (3 linear and 3 angular),  
        robot velocity (3 linear and 3 angular),  robot position(3 linear and 3 angular),  time, sigma values(3 linear and 3 angular),  
        mode (A or B), whether the subject is moving towards singularity or away from it (1 linear and 1 angular).
        """

        ob_statuss = self.inlet_robot.pull_sample()[0]
        for i in range(0,299):
            ob_statuss=np.append(ob_statuss,self.inlet_robot.pull_sample()[0])

        print(ob_statuss)
        ob_status=ob_statuss
        print(ob_status)
        print(np.shape(ob_status))
        # Magnitude Force
        mag_force = np.sqrt(np.square(ob_status[6]) * np.square(ob_status[7]) * np.square(ob_status[8]))
        print('Mag F '+ str(mag_force))
        # Magnitude Force for rotation
        mag_force_rot = np.sqrt(np.square(ob_status[9]) * np.square(ob_status[10]) * np.square(ob_status[11]))
        print('Mag RF ' + str(mag_force_rot))

        # Magnitude Velocity
        mag_vel = np.sqrt(np.square(ob_status[12]) * np.square(ob_status[13]) * np.square(ob_status[14]))
        print('Mag V ' + str(mag_vel))

        # Magnitude Velocity for rotation
        mag_vel_rot = np.sqrt(np.square(ob_status[15]) * np.square(ob_status[16]) * np.square(ob_status[17]))
        print('Mag RV ' + str(mag_force_rot))

        # Definition of environment observation_space
        min_obs = np.array([0,0,0,0]) #np.concatenate((np.zeros(np.size(mag_force)), np.zeros(np.size(mag_force_rot)),
                             #     np.zeros(np.size(mag_vel)), np.zeros(np.size(mag_vel_rot))))
        max_obs = np.array([mag_force, mag_force_rot, mag_vel, mag_vel_rot])

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _get_current_state(self):
        """Get current state of envionment.
        Returns:
            gym.spaces: Gym observation space object.
        """

        """
        Data as per Robot:
  
        const char * channels[] = {"sigma1", "sigma2", "sigma3", "sigma4", "sigma5", "sigma6", "int_force_x", "int_force_y",
                     "int_force_z", "int_force_xrot", "int_force_yrot", "int_force_zrot", "tcp_vel_x",
                     "tcp_vel_y", "tcp_vel_z", "tcp_vel_xrot", "tcp_vel_yrot", "tcp_vel_zrot", "tcp_pos_x",
                     "tcp_pos_y", "tcp_pos_z", "tcp_pos_xrot", "tcp_pos_yrot", "tcp_pos_zrot", "time", "sing1",
                     "sing2", "sing3", "sing4", "sing5", "sing6", "mode", "towordsSing1", "towordsSing2"};
  
        They are: whether or not sing avoidance is on (3 linear and 3 angular), interaction forces (3 linear and 3 angular),  
        robot velocity (3 linear and 3 angular),  robot position(3 linear and 3 angular),  time, sigma values(3 linear and 3 angular),  
        mode (A or B), whether the subject is moving towards singularity or away from it (1 linear and 1 angular).
        """

        ob_status = self.inlet_robot.pull_sample()[0]

        # Magnitude Force
        mag_force = np.sqrt(np.square(ob_status[6]) * np.square(ob_status[7]) * np.square(ob_status[8]))

        # Magnitude Force for rotation
        mag_force_rot = np.sqrt(np.square(ob_status[9]) * np.square(ob_status[10]) * np.square(ob_status[11]))

        # Magnitude Velocity
        mag_vel = np.sqrt(np.square(ob_status[12]) * np.square(ob_status[13]) * np.square(ob_status[14]))

        # Magnitude Velocity for rotation
        mag_vel_rot = np.sqrt(np.square(ob_status[15]) * np.square(ob_status[16]) * np.square(ob_status[17]))

        return np.array([mag_force, mag_force_rot, mag_vel, mag_vel_rot])

    def render(self, mode='human', close=False):
        return 0

# print statement or ignore
