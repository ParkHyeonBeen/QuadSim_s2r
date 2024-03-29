import numpy as np
from numpy import linalg
from envs.quadrotor import QuadRotorAsset
from envs.utils_env import *
import gym
from envs.pid_controller import PIDController

class Sim2RealEnv(QuadRotorAsset):
    def __init__(
            self,
            args,
            reset_noise_scale=0.1
    ):
        super(Sim2RealEnv, self).__init__(args=args)

        self.env_render = gym.make("QuadRotor-v0")
        self.env_render.reset()
        self.args = args
        self._reset_noise_scale = reset_noise_scale
        self.suc_cnt = 0
        self.drop_cnt = 0
        if self.args.train:
            self.goal = np.array([0., 0., 2.])
        else:
            self.goal = np.array(args.set_goal)
        self.init_max_pbox = args.init_max_pbox
        self.init_max_ang = args.init_max_ang
        self.init_max_vel = args.init_max_vel
        self.init_max_ang_vel = args.init_max_ang_vel

        self.random_ratio = args.random_ratio
        self.random_min = 1. - self.random_ratio
        self.random_max = 1. + self.random_ratio
        self.room_size = 5.
        self.dimension = 3
        self.dense_reward = True

        self.position_dim = 3
        self.velocity_dim = 3
        self.rotation_dim = 6
        self.angular_velocity_dim = 3
        self.state_dim = self.position_dim + self.velocity_dim + self.rotation_dim + self.angular_velocity_dim
        self.action_dim = 4
        self.n_history = args.n_history
        self.u_hat = 0.

        self.thrust_noise_sigma = args.thrust_noise_sigma
        self.thrust_noise = OUNoise(self.action_dim, sigma=self.thrust_noise_sigma)

        self.controller = PIDController(args)
        self.dist_scale = 0.0

    def reset(self):
        self.local_step = 0
        self.state = np.zeros(12)

        if not self.args.train:
            # Randomize initial states
            self.state[:3] = self.goal + self.init_max_pbox * (np.random.randint(2, size=3)*2 - 1) / np.array(
                [1, 1, 2])
            if self.args.set_path == 'sine':
                self.init_pos = self.state[:3].copy()

            self.state[3:6] = np.pi*(np.random.randint(2, size=3)*2 - 1)/ np.array(
                [180 / self.init_max_ang, 180 / self.init_max_ang, 1])
            self.state[6:9] = self.init_max_vel*(np.random.randint(2, size=3)*2 - 1)
            self.state[9:] = self.init_max_ang_vel*(np.random.randint(2, size=3)*2 - 1)

            # Randomize parameter of the quadrotor
            self.mass = self.init_mass * (1. + self.random_ratio)
            self.length = self.init_length * (1. + self.random_ratio)
            self.inertia = self.init_inertia * (1. + self.random_ratio)
            # self.kt = self.init_kt * (1. + self.random_ratio)
            # self.lag_ratio = self.init_lag_ratio * (1. + self.random_ratio)
        else:
            # Randomize initial states
            self.state[:3] = self.goal + np.random.uniform(-self.init_max_pbox, self.init_max_pbox, 3) / np.array([1, 1, 2])
            self.state[3:6] = np.random.uniform(-np.pi, np.pi, 3) / np.array([180/self.init_max_ang, 180/self.init_max_ang, 1])
            self.state[6:9] = np.random.uniform(-self.init_max_vel, self.init_max_vel, 3)
            self.state[9:] = np.random.uniform(-self.init_max_ang_vel, self.init_max_ang_vel, 3)

            # Randomize parameter of the quadrotor
            self.mass = self.init_mass * np.random.uniform(self.random_min, self.random_max)
            self.length = self.init_length * np.random.uniform(self.random_min, self.random_max)
            self.kt = self.init_kt * np.random.uniform(self.random_min, self.random_max)
            self.inertia = self.init_inertia * np.random.uniform(self.random_min, self.random_max)
            self.lag_ratio = self.init_lag_ratio * np.random.uniform(self.random_min, self.random_max)

        # PID controller
        self.controller.set_state(self.state[:6], self.state[6:])

        obs, _ = self._get_obs(mem_reset=True)
        return obs

    def step(self, action_tanh):

        self.local_step += 1
        if self.local_step % 5 == 0:
            self._make_path()

        # RL policy controller
        self.action_tanh = action_tanh
        f_hat = (action_tanh + 1) / 2
        u_hat = np.sqrt(np.clip(f_hat, 0, 1))  # rotor angular velocity

        # motor lag
        u_hat = self.lag_ratio*(u_hat - self.u_hat) + self.u_hat
        self.u_hat = u_hat

        # motor noise
        noise = self.thrust_noise.noise()
        f = np.clip((u_hat+noise) ** 2, 0., 1.)

        # Normalized force to real force
        fmax = self.kt * self.max_rpm ** 2
        f *= fmax

        # PID controller
        # f = self.controller.get_force(self.state, self._get_obs()[0])
        #
        if not self.args.train:
            f = add_disturbance(f, fmax, self.local_step, self.args.episode_length, scale=self.dist_scale, frequency=4)

        self.do_simulation(f)
        new_obs, done = self._get_obs(mem_reset=False)
        reward = self._get_reward(new_obs, done)
        info = None

        return new_obs, reward, done, info, f

    def _get_reward(self, obs, done):
        Ep, Ev, Ew = obs["position_error_obs"][:self.position_dim], \
                     obs["velocity_error_obs"][:self.velocity_dim], \
                     obs["angular_velocity_error_obs"][:self.angular_velocity_dim]
        R = obs["rotation_obs"][:self.rotation_dim].reshape([3, 2])
        action = obs["action_obs"][:self.action_dim]

        dist = np.linalg.norm(Ep)
        # dist = np.linalg.norm(Ep[:2])
        # dist_h = np.abs(Ep[-1])

        loss_phi = -1. * R[0, 1]
        loss_tht = -1. * R[1, 1]
        loss_yaw = -1. * R[2, 1]

        r_pos = np.exp(-dist / 2)
        # r_h = np.exp(-dist_h / 2)

        r_roll = (1 - loss_phi) / 2
        r_pitch = (1 - loss_tht) / 2
        r_yaw = (1 - loss_yaw) / 2
        # r_yaw = np.exp((-1 - loss_yaw)*2)

        r_vel = (1 + np.exp(-(np.linalg.norm(Ev) ** 2) * np.log(10) / 25)) / 2
        r_angvel = (3 + np.exp(-(np.linalg.norm(Ew) ** 2) * np.log(10) / 25)) / 4

        r_action = np.mean(np.exp(-(action + 1) / 2))

        reward = (r_vel * r_angvel + 5 * r_pos * r_roll * r_pitch * r_yaw) / 6
        # reward = (2 * r_vel * r_angvel +
        #           7 * r_pos * r_h * r_roll * r_pitch * r_yaw +
        #           r_action) / 10

        return reward

    def reward_func(self, transition):
        Ep, Ev, Ew = transition["position_error_obs"][:, :self.position_dim], \
                     transition["velocity_error_obs"][:, :self.velocity_dim], \
                     transition["angular_velocity_error_obs"][:, :self.angular_velocity_dim]
        R = transition["rotation_obs"][:, :self.rotation_dim].reshape([-1, 3, 2])
        action = transition["action_obs"][:, :self.action_dim]

        dist = np.linalg.norm(Ep, axis=1)
        # dist = np.linalg.norm(Ep[:2], axis=1)
        # dist_h = np.abs(Ep[-1])

        loss_phi = -1. * R[:, 0, 1]
        loss_tht = -1. * R[:, 1, 1]
        loss_yaw = -1. * R[:, 2, 1]

        r_pos = np.exp(-dist / 2)
        # r_h = np.exp(-dist_h / 2)

        r_roll = (1 - loss_phi) / 2
        r_pitch = (1 - loss_tht) / 2
        r_yaw = (1 - loss_yaw) / 2
        # r_yaw = np.exp((-1 - loss_yaw)*2)

        r_vel = (1 + np.exp(-(np.linalg.norm(Ev, axis=1) ** 2) * np.log(10) / 25)) / 2
        r_angvel = (3 + np.exp(-(np.linalg.norm(Ew, axis=1) ** 2) * np.log(10) / 25)) / 4

        r_action = np.mean(np.exp(-(action + 1) / 2), axis=1)

        reward = (r_vel * r_angvel + 5 * r_pos * r_roll * r_pitch * r_yaw) / 6
        # reward = (2 * r_vel * r_angvel +
        #           7 * r_pos * r_h * r_roll * r_pitch * r_yaw +
        #           r_action) / 10

        return reward

    def _get_obs(self, mem_reset=False):
        if self.n_history == 1:
            mem_reset = True
        position = self.state[:3]
        velocity = self.state[6:9]
        angular_velocity = self.state[9:12]

        error_state = self._get_error_state(position, self.goal, velocity, angular_velocity)
        position_error = error_state[0]
        velocity_error = error_state[1]
        rotation = self._get_euler(self.state[3:6]).reshape([-1])
        angular_velocity_state = error_state[2]

        # Get collision info. (True / False)
        crashed = position[-1] - 4. >= -self.length
        crashed = crashed or not np.array_equal(position,
                                                np.clip(position, a_min=-self.room_size, a_max=self.room_size))
        crashed = crashed or (abs(self.state[3]) > np.pi / 2 or abs(self.state[4]) > np.pi / 2)
        done = crashed

        # Update history
        if mem_reset:  # first time step
            self.mem_position = np.concatenate([position] * self.n_history)
            self.mem_velocity = np.concatenate([velocity] * self.n_history)
            self.mem_position_error = np.concatenate([position_error] * self.n_history)
            self.mem_velocity_error = np.concatenate([velocity_error] * self.n_history)
            self.mem_rotation = np.concatenate([rotation] * self.n_history)
            self.mem_angular_velocity = np.concatenate([angular_velocity] * self.n_history)
            self.mem_angular_velocity_error = np.concatenate([angular_velocity_state] * self.n_history)
            self.mem_action = np.concatenate([np.zeros(self.action_dim)] * self.n_history, axis=-1)
        else:
            self.mem_position[len(position):] = self.mem_position[:len(position) * (self.n_history - 1)]
            self.mem_position[:len(position)] = position
            self.mem_velocity[len(velocity):] = self.mem_velocity[:len(velocity) * (self.n_history - 1)]
            self.mem_velocity[:len(velocity)] = velocity
            self.mem_position_error[len(position_error):] = self.mem_position_error[
                                                            :len(position_error) * (self.n_history - 1)]
            self.mem_position_error[:len(position_error)] = position_error
            self.mem_velocity_error[len(velocity_error):] = self.mem_velocity_error[
                                                            :len(velocity_error) * (self.n_history - 1)]
            self.mem_velocity_error[:len(velocity_error)] = velocity_error
            self.mem_rotation[len(rotation):] = self.mem_rotation[:len(rotation) * (self.n_history - 1)]
            self.mem_rotation[:len(rotation)] = rotation
            self.mem_angular_velocity_error[len(angular_velocity_state):] = self.mem_angular_velocity_error[
                                                                            :len(angular_velocity_state) * (
                                                                                        self.n_history - 1)]
            self.mem_angular_velocity_error[:len(angular_velocity_state)] = angular_velocity_state
            self.mem_angular_velocity[len(angular_velocity):] = self.mem_angular_velocity[
                                                                :len(angular_velocity) * (self.n_history - 1)]
            self.mem_angular_velocity[:len(angular_velocity)] = angular_velocity
            self.mem_action[len(self.action_tanh):] = self.mem_action[:len(self.action_tanh) * (self.n_history - 1)]
            self.mem_action[:len(self.action_tanh)] = self.action_tanh

        # Observation
        obs = {}
        obs["position_obs"] = self.mem_position.copy()
        obs["position_error_obs"] = self.mem_position_error.copy()
        obs["velocity_error_obs"] = self.mem_velocity_error.copy()
        obs["rotation_obs"] = self.mem_rotation.copy()
        obs["angular_velocity_error_obs"] = self.mem_angular_velocity_error.copy()
        obs["action_obs"] = self.mem_action.copy()

        return obs, done

    def _get_error_state(self, curr_pos_, goal_pos_, vel_, ang_vel_, history=False):
        if history:  # when sample data from replay buffer, used only HER
            # TARGET
            tar_xyz = np.expand_dims(goal_pos_, axis=1)
            tar_xyz_vel = np.zeros((3, 1))
            tar_ang_vel = np.zeros((3, 1))
            # CURRENT
            xyz = curr_pos_.reshape([curr_pos_.shape[0], self.n_history, -1])
            vel = vel_.reshape([vel_.shape[0], self.n_history, -1])
            ang_vel = ang_vel_.reshape([ang_vel_.shape[0], self.n_history, -1])
            # ERROR
            err_xyz = tar_xyz - xyz
            err_xyz_vel = tar_xyz_vel - vel
            err_ang_vel = tar_ang_vel - ang_vel
            # OBSERVATION
            # error_state = np.concatenate([err_xyz, err_xyz_vel, err_ang_vel])
            error_state = err_xyz
            error_state = error_state.reshape([curr_pos_.shape[0], -1])

        else:  # when drone interact in environment
            # TARGET
            tar_xyz = np.reshape(goal_pos_, [-1, 3])
            tar_xyz_vel = np.zeros((1, 3))
            tar_ang_vel = np.zeros((1, 3))
            # CURRENT
            xyz = np.reshape(curr_pos_, [-1, 3])
            vel = np.reshape(vel_, [-1, 3])
            ang_vel = np.reshape(ang_vel_, [-1, 3])
            # ERROR
            err_xyz = tar_xyz - xyz
            err_xyz_vel = tar_xyz_vel - vel
            err_ang_vel = tar_ang_vel - ang_vel
            # OBSERVATION
            error_state = np.concatenate([err_xyz, err_xyz_vel, err_ang_vel])

        return error_state

    def _make_path(self, radius=2.):
        if self.args.set_path == 'circle':
            self.goal[0] = radius * math.cos((2*math.pi / self.args.episode_length) * self.local_step)
            self.goal[1] = radius * math.sin((2*math.pi / self.args.episode_length) * self.local_step)

        if self.args.set_path == 'helix':
            self.goal[0] = radius * math.cos((2*math.pi / self.args.episode_length) * self.local_step)
            self.goal[1] = radius * math.sin((2*math.pi / self.args.episode_length) * self.local_step)
            self.goal[2] = 5 * (self.local_step / self.args.episode_length)

        if self.args.set_path == 'sine':
            pass

    """ relted to render """

    def render(self):
        pos = np.array([self.state[1], self.state[0], -self.state[2]])
        quat = np.array(rpy2quat(self.state[4], self.state[3], -self.state[5]))
        qpos = np.concatenate((pos, quat))
        qvel = np.array(
            [self.state[7], self.state[6], -self.state[8], self.state[10], self.state[9], -self.state[11]])
        self.env_render.unwrapped.set_state(qpos, qvel)
        self.env_render.render()

if __name__ == "__main__":
    env = Sim2RealEnv()
    env.reset()
    ctrl = np.array([0., 0., 0., 0.])
    for _ in range(1000):
        obs, _, done, _, _ = env.step(ctrl)
        print(obs["position_obs"], done)
        env.render()
        time.sleep(0.03)

