import numpy
import numpy as np
import pandas as pd
import gym
import random, math
import torch
import torch.nn as nn
from collections import deque
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
import sys, os, time
from pathlib import Path
from IPython.display import clear_output

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


sys.path.append(str(Path('Utils.py').parent.absolute()))  # 절대 경로에 추가

class DataManager:
    def __init__(self, path_dim=2, data_name=None):
        self.data = None
        self.data_name = data_name
        self.path_dim = path_dim
        self.path_data = np.empty([1, path_dim])

    def init_data(self):
        self.data = None

    def put_data(self, obs):
        if self.data is None:
            self.data = obs
        else:
            self.data = np.vstack((self.data, obs))

    def put_path(self, obs):
        self.path_data = np.vstack((self.path_data, obs[:3]))

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
        plt.show(block=False)
        plt.pause(0.0001)
        plt.cla()

    def plot_path(self, obs, label=None):
        self.put_path(obs)
        if label is None:
            plt.plot(self.path_data[:, i] for i in range(self.path_dim))
        else:
            plt.plot([self.path_data[:, i] for i in range(self.path_dim)], label=label)
            plt.legend()
        plt.show(block=False)
        plt.pause(0.0001)
        plt.cla()

    def save_data(self, path, fname, numpy=False):

        if numpy is False:
            df = pd.DataFrame(self.data)
            df.to_csv(path + fname + ".csv")
        else:
            df = np.array(self.data)
            np.save(path + fname + ".npy", df)

    def save_path(self, path, fname, numpy=False):
        if numpy is False:
            df = pd.DataFrame(self.data)
            df.to_csv(path + fname + ".csv")
        else:
            df = np.array(self.path_data)
            np.save(path + fname + ".npy", df)

    def plot_fig(self, path):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.plot(np.arange(len(self.data)), self.data)
        plt.grid(True)
        plt.savefig(path)
        # plt.show()
        plt.clf()

    def plot_variance_fig(self, path):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        mean_val = self.data[:, 0]
        std_val = self.data[:, 1]
        x = range(len(mean_val))
        plt.plot(x, mean_val)
        y1 = np.asarray(mean_val) + np.asarray(std_val)
        y2 = np.asarray(mean_val) - np.asarray(std_val)
        plt.fill_between(x, y1, y2, alpha=0.3)
        plt.grid(True)
        plt.savefig(path)
        # plt.show()
        plt.clf()


def plot_fig(data, path):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    x = range(len(data))
    plt.plot(x, data)
    plt.grid(True)
    plt.savefig(path)
    # plt.show()
    plt.clf()

def plot_variance_fig(mean_val, std_val, path):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    x = range(len(mean_val))
    plt.plot(x, mean_val)
    y1 = np.asarray(mean_val) + np.asarray(std_val)
    y2 = np.asarray(mean_val) - np.asarray(std_val)
    plt.fill_between(x, y1, y2, alpha=0.3)
    plt.grid(True)
    plt.savefig(path)
    # plt.show()
    plt.clf()

## related to control ##
def quat2mat(quat):
    """ Convert Quaternion to Rotation matrix.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def rot2rpy(R):
    temp = np.array([0, 0, 1]) @ R
    pitch = math.asin(-temp[0])
    roll = math.asin(temp[1] / math.cos(pitch))
    yaw = math.acos(R[0, 0] / math.cos(pitch))

    return roll, pitch, yaw


def quat2rpy(quat):
    R = quat2mat(quat)
    euler = rot2rpy(R)
    euler = np.array(euler)
    return euler

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


def inv_softsign(y):
    if type(y) is torch.Tensor:
        y = y.cpu().detach().numpy()

    x = np.where(y >= 0, y/(1-y), y/(1+y))
    return x


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

## related to saved data ##

def create_save_dir(args, algorithm_name):

    root = args.path + 'storage'
    if not os.path.isdir(root): os.mkdir(root)

    env_dir = root + "/" + args.env_name
    if not os.path.isdir(env_dir): os.mkdir(env_dir)

    if args.ensemble_mode is True:
        save_dir = env_dir + "/%s_" % time.strftime("%m%d") + algorithm_name + "_ensemble"
    else:
        save_dir = env_dir + "/%s_" % time.strftime("%m%d") + algorithm_name
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    log_dir = save_dir + "/saved_log"
    net_dir = save_dir + "/saved_net"
    if not os.path.isdir(log_dir): os.mkdir(log_dir)
    if not os.path.isdir(net_dir): os.mkdir(net_dir)

    log_eval_dir = log_dir + "/eval"
    log_evalby_dir = log_dir + "/eval_by"
    if not os.path.isdir(log_eval_dir): os.mkdir(log_eval_dir)
    if not os.path.isdir(log_evalby_dir): os.mkdir(log_evalby_dir)

    policy_dir = net_dir + "/policy"
    if not os.path.isdir(policy_dir): os.mkdir(policy_dir)

    model_dir = net_dir + "/model"
    dnn_dir = model_dir + "/DNN"
    bnn_dir = model_dir + "/BNN"
    etc_dir = model_dir + "/Etc"
    if args.modelbased_mode is True:
        if not os.path.isdir(model_dir): os.mkdir(model_dir)
        if not os.path.isdir(dnn_dir): os.mkdir(dnn_dir)
        if not os.path.isdir(bnn_dir): os.mkdir(bnn_dir)
        if not os.path.isdir(etc_dir): os.mkdir(etc_dir)

    paths = {'save': save_dir,
             'log': log_dir,
             'log_eval': log_eval_dir,
             'log_evalby': log_evalby_dir,
             'model': model_dir,
             'policy': policy_dir,
             'dnn': dnn_dir,
             'bnn': bnn_dir,
             'etc': etc_dir,
             }

    return paths

def np2str(nump):
    _str = ""
    for element in nump:
        _str += (str(element) + " ")
    return _str

def create_config(algorithm_name, args, env, state_dim, action_dim, max_action, min_action):

    max_action_str = np2str(max_action)
    min_action_str = np2str(min_action)

    with open(args.path + 'config.txt', 'w') as f:
        print("Develop mode:", args.develop_mode, file=f)
        print("Environment:", args.env_name, file=f)
        print("Algorithm:", algorithm_name, file=f)
        print("State dim:", state_dim, file=f)
        print("Action dim:", action_dim, file=f)
        print("Max action:", max_action_str, file=f)
        print("Min action:", min_action_str, file=f)
        print("Step size:", env.env.dt, file=f)
        print("Frame skip:", env.env.frame_skip, file=f)
        print("Save path :", args.path, file=f)
        print("Model based mode:", args.modelbased_mode, file=f)
        print("model lr : {}, model klweight : {}, inv model lr dnn: {}, inv model lr bnn: {}, inv model klweight : {}".
              format(args.model_lr, args.model_kl_weight, args.inv_model_lr_dnn, args.inv_model_lr_bnn, args.inv_model_kl_weight), file=f)
        print("consideration note : ", args.note, file=f)
        print(" ")

def load_config(args):

    if args.prev_result is True:
        path_config = args.path + "storage/_prev/trash/" + args.prev_result_fname + "/config.txt"
        path_policy = args.path + "storage/_prev/trash/" + args.prev_result_fname + "/saved_net/policy/" + args.policynet_name
    else:
        path_config = args.path + args.result_index + "config.txt"
        path_policy = args.path + args.result_index + "saved_net/policy/" + args.policynet_name

    modelbased_mode_cfg = False
    ensemble_mode_cfg = False

    with open(path_config, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # print(line[:len(line)-1])
            if 'Environment:' in line:
                env_name_cfg = line[line.index(':')+2:len(line)-1]
            if 'Algorithm:' in line:
                algorithm_cfg = line[line.index(':')+2:len(line)-1]
            if 'State dim:' in line:
                state_dim_cfg = int(line[line.index(':')+2:len(line)-1])
            if 'Action dim:' in line:
                action_dim_cfg = int(line[line.index(':')+2:len(line)-1])
            if 'Max action:' in line:
                max_action_cfg = np.fromstring(line[line.index(':')+2:len(line)-1], dtype=float, sep=" ")
            if 'Min action:' in line:
                min_action_cfg = np.fromstring(line[line.index(':')+2:len(line)-1], dtype=float, sep=" ")
            if 'Frame skip:' in line:
                frame_skip_cfg = int(line[line.index(':')+2:len(line)-1])
            if 'Model based mode:' in line:
                modelbased_mode_cfg = (line[line.index(':')+2:len(line)-1] == 'True')
            if 'Ensemble mode:' in line:
                ensemble_mode_cfg = (line[line.index(':')+2:len(line)-1] == 'True')

    return path_policy, env_name_cfg, algorithm_cfg, state_dim_cfg, action_dim_cfg, max_action_cfg, min_action_cfg, \
           frame_skip_cfg, modelbased_mode_cfg, ensemble_mode_cfg


def get_algorithm_info(algorithm_name, state_dim, action_dim, device):

    # print(algorithm_name)
    # print('SAC_v2')
    # print(algorithm_name == 'SAC_v2')

    if algorithm_name == 'SAC_v2':
        from Example.run_SACv3 import hyperparameters
        from Algorithm.SAC_v3 import SAC_v2
        _args = hyperparameters()
        _algorithm = SAC_v2(state_dim, action_dim, device, _args)
    elif algorithm_name == 'DDPG':
        from Example.run_DDPG import hyperparameters
        from Algorithm.DDPG import DDPG
        _args = hyperparameters()
        _algorithm = DDPG(state_dim, action_dim, device, _args)
    else:
        raise Exception("check the name of algorithm")
    return _args, _algorithm

def save_policys(policy, score, score_now, alive_rate, path):

    if score_now > score and alive_rate > 0.9:
        torch.save(policy.actor.state_dict(), path + "saved_net/policy/policy_best")
        torch.save(policy.pid.state_dict(), path + "saved_net/policy/pid_best")
        return score_now
    elif score_now > score:
        torch.save(policy.actor.state_dict(), path + "saved_net/policy/policy_better")
        torch.save(policy.pid.state_dict(), path + "saved_net/policy/pid_better")
        return score_now
    else:
        torch.save(policy.actor.state_dict(), path + "saved_net/policy/policy_current")
        torch.save(policy.pid.state_dict(), path + "saved_net/policy/pid_current")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')

## related to gym

def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    random.seed(random_seed)

    return random_seed

def gym_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def suite_env(*arg):
    import robosuite as suite

    # robosuite
    env = suite.make(env_name=arg[0],
                     robots=arg[1],
                     has_renderer=arg[2],
                     has_offscreen_renderer=arg[3],
                     use_camera_obs=arg[4],
                     reward_shaping=True)

    test_env = suite.make(env_name=arg[0],
                     robots=arg[1],
                     has_renderer=arg[2],
                     has_offscreen_renderer=arg[3],
                     use_camera_obs=arg[4],
                     reward_shaping=True)

    return env, test_env

def obs_to_state(env, obs):
    state = None
    for key in env.active_observables:
        if state is None:
            state = obs[key]
        else:
            state = np.hstack((state, obs[key]))
    return state

def soft_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

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

def copy_weight(network, target_network):
    target_network.load_state_dict(network.state_dict())

def tie_conv(source, target):
    #source ->  target
    for i in range(source.layer_num):
        target.conv[i].weight = source.conv[i].weight
        target.conv[i].bias = source.conv[i].bias

def atari_env(env_name, image_size, frame_stack, frame_skip, random_seed):
    import gym
    from gym.wrappers import AtariPreprocessing, FrameStack
    env = gym.make(env_name)
    env = AtariPreprocessing(env, frame_skip=frame_skip, screen_size=image_size, grayscale_newaxis=True)
    env = FrameStack(env, frame_stack)

    env._max_episode_steps = 10000
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env = AtariPreprocessing(test_env, frame_skip=frame_skip, screen_size=image_size,
                                  grayscale_newaxis=True)
    test_env._max_episode_steps = 10000
    test_env = FrameStack(test_env, frame_stack)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def dmc_env(env_name, random_seed):
    import dmc2gym
    # deepmind control suite
    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)
    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed)

    return env, test_env

def dmc_image_env(env_name, image_size, frame_stack, frame_skip, random_seed):
    import dmc2gym
    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False,
                       from_pixels=True, height=image_size, width=image_size,
                       frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    env = FrameStack(env, k=frame_stack)

    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True, height=image_size, width=image_size,
                            frame_skip=frame_skip)  # Pre image size for curl, image size for dbc
    test_env = FrameStack(test_env, k=frame_stack)

    return env, test_env

def dmcr_env(env_name, image_size, frame_skip, random_seed, mode='classic'):
    assert mode in {'classic', 'generalization', 'sim2real'}

    import dmc_remastered as dmcr

    domain_name = env_name.split('/')[0]
    task_name = env_name.split('/')[1]
    if mode == 'classic':#loads a training and testing environment that have the same visual seed
        env, test_env = dmcr.benchmarks.classic(domain_name, task_name, visual_seed=random_seed, width=image_size, height=image_size, frame_skip=frame_skip)
    elif mode == 'generalization':#creates a training environment that selects a new visual seed from a pre-set range after every reset(), while the testing environment samples from visual seeds 1-1,000,000
        env, test_env = dmcr.benchmarks.visual_generalization(domain_name, task_name, num_levels=100, width=image_size, height=image_size, frame_skip=frame_skip)
    elif mode == 'sim2real':#approximates the challenge of transferring control policies from simulation to the real world by measuring how many distinct training levels the agent needs access to before it can succesfully operate in the original DMC visuals that it has never encountered.
        env, test_env = dmcr.benchmarks.visual_sim2real(domain_name, task_name, num_levels=random_seed, width=image_size, height=image_size, frame_skip=frame_skip)

    return env, test_env

def procgen_env(env_name, frame_stack, random_seed):
    import gym
    env_name = "procgen:procgen-{}-v0".format(env_name)
    env = gym.make(env_name, render_mode='rgb_array')
    env._max_episode_steps = 1000
    env = FrameStack(env, frame_stack, data_format='channels_last')

    test_env = gym.make(env_name, render_mode='rgb_array')
    test_env._max_episode_steps = 1000
    test_env = FrameStack(test_env, frame_stack, data_format='channels_last')

    return env, test_env

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand(size=obs.shape) / bins
    obs = obs - 0.5
    return obs

def random_crop(imgs, output_size, data_format='channels_first'):#random crop for curl
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size

    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop_image(image, output_size):#center crop for curl
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]

    return image

def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, data_format='channels_first'):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape

        if data_format == 'channels_first':
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=((shp[0] * k,) + shp[1:]),
                dtype=env.observation_space.dtype
            )
            self.channel_first = True
        else:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(shp[0:-1] + (shp[-1] * k,)),
                dtype=env.observation_space.dtype
            )
            self.channel_first = False

        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        if self.channel_first == True:
            return np.concatenate(list(self._frames), axis=0)
        elif self.channel_first == False:
            return np.concatenate(list(self._frames), axis=-1)

