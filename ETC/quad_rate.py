from numpy import linalg
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np
import math

class QuadRateEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'quadrotor.xml', 2)
        utils.EzPickle.__init__(self)

    def step(self, action):

        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        reward = 0
        done = False
        info = {}

        return ob, reward, done, info

    def _get_obs(self):
        pos = self.sim.data.qpos * 1e-0
        vel = self.sim.data.qvel * 1e-0
        return np.concatenate([pos.flat, vel.flat])