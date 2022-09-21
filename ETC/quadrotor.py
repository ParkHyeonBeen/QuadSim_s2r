import numpy as np
import math
from numpy import linalg

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 12.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class QuadRotorEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 500,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(**locals())
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        MujocoEnv.__init__(
            self, 'quadrotor.xml', 2, observation_space=observation_space
        )

    def step(self, action):

        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()
        reward = 0
        done = False
        info = {}

        return ob, reward, done, info

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_model(self):
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:6], 
                self.data.qvel.flat[:6],
            ]
        )
