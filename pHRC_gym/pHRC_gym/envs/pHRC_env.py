import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time


# For LSL
import random
import time
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream #, resolve_stream_byprop



class pHRCEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self): #,streams_robot,streams_PEN,stream_sigma):

      # Init LSL streams
      # streams_robot = resolve_stream('name','RobotData')
      # print("Crete Stream 1")
      # stream_sigma= StreamInfo('MarkerStream', 'Sigma', 1, 0, 'string', 'xxx')
      # print("Crete Stream 1")
      # streams_PEN = resolve_stream('type', 'Markers', 'name','PEN') #resolve_stream('type', 'Markers')
      # print("Crete Stream 1")
      # create a new inlet to read from the stream
      self.inlet_robot = StreamInlet(streams_robot[0])

      # create a new outlet to send data to the stream
      self.outlet = StreamOutlet(stream_sigma)

      # create a new inlet to read from the stream
      self.inlet_PEN = StreamInlet(streams_PEN[0])

      # Actions Space
      self.action_space = spaces.Box(
          low=np.array([0.15, 0.25]), high=np.array([0.25, 0.45]), dtype=np.float16)

      # The observation Space
      self.observation_space =self._get_observation_space()

          # spaces.Box(
          # low=np.array([0.15, 0.25]), high=np.array([0.35, 0.45]), dtype=np.float16)

      self.SET_TIME = 20 # 20-minutes
      self.start_time = time.time()

      # Set Initial state of gym game environment


  def step(self, action):


      # Take an action -must return next state,
      # reward for current state,
      # is done or not
      # Info (could be empty

      self.outlet.push_sample(str(action))

      self.state=self._get_current_state()

      # Check if the environment state is contained in the observation space
      if not self.observation_space.contains(self.state):
          print('InvalidState')
          # raise InvalidStateError()

      # Assign reward
      rewards= self.inlet_PEN.pull_sample()

      if rewards=="1":
          reward=1
      else:
          rewards =0

      # If total time of session is completed
      if (time.time() - self.start_time)//(self.SET_TIME*60) == 0:
          done=True
      else:
          done=False

      info={}


      return self.state,reward,done,info

  def reset(self):

      self.outlet.push_sample("0.15")
      self.outlet.push_sample("0.25")
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

      ob_status = self.inlet_robot.pull_chunk()

      # Magnitude Force
      mag_force = np.sqrt(np.square(ob_status[6]) * np.square(ob_status[7]) * np.square(ob_status[8]))

      # Magnitude Force for rotation
      mag_force_rot = np.sqrt(np.square(ob_status[9]) * np.square(ob_status[10]) * np.square(ob_status[11]))

      # Magnitude Velocity
      mag_vel = np.sqrt(np.square(ob_status[12]) * np.square(ob_status[13]) * np.square(ob_status[14]))

      # Magnitude Velocity for rotation
      mag_vel_rot = np.sqrt(np.square(ob_status[15]) * np.square(ob_status[16]) * np.square(ob_status[17]))

      # Definition of environment observation_space
      min_obs = np.concatenate((np.zeros(np.size(mag_force)), np.zeros(np.size(mag_force_rot)),np.zeros(np.size(mag_vel)),np.zeros(np.size(mag_vel_rot)) ))
      max_obs = np.concatenate((mag_force,mag_force_rot,mag_vel,mag_vel_rot))

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

      ob_status = self.inlet_robot.pull_chunk()

      # Magnitude Force
      mag_force = np.sqrt(np.square(ob_status[6]) * np.square(ob_status[7]) * np.square(ob_status[8]))

      # Magnitude Force for rotation
      mag_force_rot = np.sqrt(np.square(ob_status[9]) * np.square(ob_status[10]) * np.square(ob_status[11]))

      # Magnitude Velocity
      mag_vel = np.sqrt(np.square(ob_status[12]) * np.square(ob_status[13]) * np.square(ob_status[14]))

      # Magnitude Velocity for rotation
      mag_vel_rot = np.sqrt(np.square(ob_status[15]) * np.square(ob_status[16]) * np.square(ob_status[17]))

      return np.concatenate((mag_force, mag_force_rot, mag_vel, mag_vel_rot))





  def render(self, mode='human', close=False):
      return 0

# print statement or ignore
