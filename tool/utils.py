from algorithms.source.replay_buffer import *
from envs.sim2real_v1 import *

import yaml, sys, os, math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from IPython.display import clear_output
import matplotlib.pyplot as plt

sys.path.append(str(Path('utils.py').parent.absolute()))  # 절대 경로에 추가

class DataManager:
    def __init__(self, data_name=None):
        self.data = None
        self.xticks = []
        self.data_name = data_name

    def init_data(self):
        self.data = None
        self.xticks = []

    def put_data(self, obs):
        if self.data is None:
            self.data = obs
        else:
            self.data = np.vstack((self.data, obs))

    def get_xticks(self, xtick):
        self.xticks.append(str(xtick))

    def mean_data(self):
        mean_data = np.mean(self.data, axis=0)
        return mean_data

    def plot_data(self, obs, label=None):
        self.put_data(obs)
        if label is None:
            if self.data_name is not None:
                plt.figure(self.data_name)
            plt.plot(self.data, 'o')
        else:
            plt.plot(self.data, label=label)
            plt.legend()


    def save_data(self, path, fname, numpy=False):
        if numpy is False:
            df = pd.DataFrame(self.data)
            df.to_csv(path + fname + ".csv")
        else:
            df = np.array(self.data)
            np.save(path + fname + ".npy", df)

    def plot_fig(self, path):
        clear_output(True)
        f = plt.figure(figsize=(20,5))
        plt.plot(np.arange(len(self.data)), self.data)
        plt.grid(True)
        plt.savefig(path)
        # plt.show()
        f.clf()
        plt.close(f)

    def bar_fig(self, path):
        clear_output(True)
        f = plt.figure(figsize=(20,5))
        x = np.arange(len(self.data))
        plt.bar(x, self.data.flatten())
        plt.xticks(x, self.xticks)
        plt.grid(True)
        plt.savefig(path)
        # plt.show()
        f.clf()
        plt.close(f)

    def plot_variance_fig(self, path, need_xticks=False):
        clear_output(True)
        f = plt.figure(figsize=(20,5))
        mean_val = self.data[:, 0]
        std_val = self.data[:, 1]
        x = range(len(mean_val))
        plt.plot(x, mean_val)
        y1 = np.asarray(mean_val) + np.asarray(std_val)
        y2 = np.asarray(mean_val) - np.asarray(std_val)
        plt.fill_between(x, y1, y2, alpha=0.3)
        plt.grid(True)
        if need_xticks:
            plt.xticks(x, self.xticks)
        plt.savefig(path)
        # plt.show()
        f.clf()
        plt.close(f)

def plot_fig(data, path):
    clear_output(True)
    f = plt.figure(figsize=(20,5))
    x = range(len(data))
    plt.plot(x, data)
    plt.grid(True)
    plt.savefig(path)
    # plt.show()
    f.clf()
    plt.close(f)

def plot_variance_fig(mean_val, std_val, path):
    clear_output(True)
    f = plt.figure(figsize=(20,5))
    x = range(len(mean_val))
    plt.plot(x, mean_val)
    y1 = np.asarray(mean_val) + np.asarray(std_val)
    y2 = np.asarray(mean_val) - np.asarray(std_val)
    plt.fill_between(x, y1, y2, alpha=0.3)
    plt.grid(True)
    plt.savefig(path)
    # plt.show()
    f.clf()
    plt.close(f)

def eval_plot(step, pos, vel, rpy, angvel, policy, force):
    plt.clf()
    plt.close()
    clear_output(True)
    step = np.array(range(step+2))
    rpy = rpy * 180 / math.pi

    plt.figure(1)
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.plot(step, pos[:, i])
        plt.xlim(0, 1000)
        plt.grid(True)
        if i == 0:
            plt.title("Position")

        plt.subplot(3, 2, 2 * i + 2)
        plt.plot(step, vel[:, i])
        plt.xlim(0, 1000)
        plt.grid(True)
        if i == 0:
            plt.title("Velocity")

    plt.figure(2)
    for i in range(3):
        plt.subplot(3, 2, 2 * i + 1)
        plt.plot(step, rpy[:, i])
        plt.xlim(0, 1000)
        plt.grid(True)
        if i == 0:
            plt.title("Euler angle")

        plt.subplot(3, 2, 2 * i + 2)
        plt.plot(step, angvel[:, i])
        plt.xlim(0, 1000)
        plt.grid(True)
        if i == 0:
            plt.title("Angular velocity")

    plt.figure(3, figsize=(20, 5))
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.xlim(0, 1000)
        plt.ylim(-1, 1)
        plt.plot(step, policy[:, i])
        plt.grid(True)
        if i == 0:
            plt.title("Policy action")

    plt.figure(4, figsize=(20, 5))
    for i in range(4):
        plt.subplot(4, 1, i + 1)
        plt.xlim(0, 1000)
        plt.ylim(0, 10)
        plt.plot(step, force[:, i])
        plt.grid(True)
        if i == 0:
            plt.title("Motor force")

    # plt.savefig(path + '/sac_v2_multi.png')
    plt.show()


class EnvMaker:
    def __init__(self,
                 env_name,
                 args
                 ):

        self.env_name = env_name
        if env_name == "QuadRotor-v0":
            self.env = Sim2RealEnv(args=args)
            self.position_dim = self.env.position_dim
            self.velocity_dim = self.env.velocity_dim
            self.rotation_dim = self.env.rotation_dim
            self.angular_velocity_dim = self.env.angular_velocity_dim
            self.state_dim = self.env.state_dim
            self.action_dim = self.env.action_dim
            self.replay_buffer = ReplayBuffer

        else:
            self.env = gym.make(env_name)
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            self.replay_buffer = ReplayBufferGym

    def reset(self):
        self.env.reset()

    def step(self, action):
        self.env.step(action)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def read_config(path):
    """
    Return python dict from .yml file.
    Args:
        path (str): path to the .yml config.
    Returns (dict): configuration object.
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg


def empty_torch_queue(q):
    while True:
        try:
            o = q.get_nowait()
            del o
        except:
            break
    q.close()


def share_parameters(adamoptim):
    ''' share parameters of Adamoptimizers for multiprocessing '''
    for group in adamoptim.param_groups:
        for p in group['params']:
            state = adamoptim.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()

def np2str(nump):
    _str = ""
    for element in nump:
        _str += (str(element) + " ")
    return _str

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def rot2rpy(R):
    temp = np.array([0, 0, 1]) @ R
    pitch = math.asin(-temp[0])
    roll = math.asin(temp[1] / math.cos(pitch))
    yaw = math.acos(R[0, 0] / math.cos(pitch))

    return roll, pitch, yaw


def normalize(input, act_max, act_min):

    if type(input) is not torch.Tensor:
        normal_mat = np.zeros((len(input), len(input)))
        np.fill_diagonal(normal_mat,  2 / (act_max - act_min))
    else:
        act_max = torch.tensor(act_max, dtype=torch.float).cuda()
        act_min = torch.tensor(act_min, dtype=torch.float).cuda()
        normal_mat = torch.diag(2 / (act_max - act_min))
    normal_bias = (act_max + act_min) / 2
    input = (input - normal_bias) @ normal_mat
    return input

def denormalize(input, act_max, act_min):

    if type(input) is not torch.Tensor:
        denormal_mat = np.zeros((len(input), len(input)))
        np.fill_diagonal(denormal_mat, (act_max - act_min) / 2)
    else:
        act_max = torch.tensor(act_max, dtype=torch.float).cuda()
        act_min = torch.tensor(act_min, dtype=torch.float).cuda()
        denormal_mat = torch.diag((act_max - act_min) / 2)

    denormal_bias = (act_max + act_min) / 2
    input = input @ denormal_mat + denormal_bias

    return input

def add_noise(val, scale = 0.1):
    val += scale*np.random.normal(size=len(val))
    return val


def add_disturbance(val, step, terminal_time, scale=0.1, frequency=4):
    for i in range(len(val)):
        val[i] += scale*math.sin((frequency*math.pi / terminal_time)*step)
    if scale > 0.01:
        if type(val) is torch.Tensor:
            val += 0.01*torch.normal(mean=torch.zeros_like(val), std=torch.ones_like(val))
        else:
            val += 0.01*np.random.normal(size=len(val))
    return val


def save_policy(policy, score_best, score_now, alive_rate, path):

    if score_now > score_best and alive_rate > 0.9:
        torch.save(policy.state_dict(), path + "/policy_best")
        return score_now
    elif score_now > score_best:
        torch.save(policy.state_dict(), path + "/policy_better")
        return score_now
    else:
        torch.save(policy.state_dict(), path + "/policy_current")

def load_model(network, path, fname):
    if "bnn" in fname:
        model_tmp = torch.load(path +'/' + fname)
        for key in model_tmp.copy().keys():
            if 'eps' in key:
                del (model_tmp[key])
        network.load_state_dict(model_tmp)
    else:
        network.load_state_dict(torch.load(path +'/' + fname))
    network.eval()

