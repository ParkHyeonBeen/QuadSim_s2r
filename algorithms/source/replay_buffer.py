import random, torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class ReplayBuffer:
    def __init__(self, args,  env):
        # self.env = env
        self.args = args
        self.capacity = self.args.buffer_size
        self.future_p = self.args.future_p
        self.buffer_data = []
        self.position = 0
        self._episode_num = 0
        self.num_experiences = 0
        self.n_history = args.n_history
        self.HER = self.args.HER

        episode_length = self.args.episode_length
        self.position_dim = env.position_dim * self.n_history
        self.velocity_dim = env.velocity_dim * self.n_history
        self.rotation_dim = env.rotation_dim * self.n_history
        self.angular_velocity_dim = env.angular_velocity_dim * self.n_history
        self.action_dim = env.action_dim
        self.prev_action_dim = self.action_dim * self.n_history
        self.size = int(self.args.buffer_size / episode_length)
        self._error_state_func = env._get_error_state
        self._reward_func = env.reward_func

        self.buffer = {"position_obs": np.empty([self.size, episode_length + 1] + [self.position_dim]),
                       "position_error_obs": np.empty([self.size, episode_length + 1] + [self.position_dim]),
                       "velocity_error_obs": np.empty([self.size, episode_length + 1] + [self.velocity_dim]),
                       "rotation_obs": np.empty([self.size, episode_length + 1] + [self.rotation_dim]),
                       "angular_velocity_error_obs": np.empty([self.size, episode_length + 1] + [self.angular_velocity_dim]),
                       "action_obs": np.empty([self.size, episode_length + 1] + [self.prev_action_dim]),
                       "action": np.empty([self.size, episode_length] + [self.action_dim]),
                       "done": np.empty([self.size, episode_length, 1]),
                       "length": np.empty([self.size, 1], dtype=np.int)}

    def add(self, episode_batch):
        """ Store an episodic data.

        :arg (dict) episode_batch  : An episodic sequence of data.
        """
        idx = self._episode_num % self.size
        data_n = len(episode_batch["action"])
        if data_n > 0:
            episode_batch['length'] = np.array([data_n])
            for key in self.buffer.keys():
                data = episode_batch[key]
                if key != "length":
                    self.buffer[key][idx][:len(data)] = data
                else:
                    self.buffer[key][idx] = data
            self._episode_num += 1
            self.num_experiences += data_n

    def get_batch(self, batch_size):
        """ Sample batch with HER.

        :arg (int) batch_size        : The number samples in a batch.
        :return (dict) transitions   : Sampled batch.
        """
        episode_num = min(self._episode_num, self.size)
        temp_buffer = {}


        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][:episode_num]

        # Construct next_state (s') data.
        temp_buffer["position_next_obs"] = self.buffer["position_obs"][:, 1:, :]
        temp_buffer["position_error_next_obs"] = self.buffer["position_error_obs"][:, 1:, :]
        temp_buffer["velocity_error_next_obs"] = self.buffer["velocity_error_obs"][:, 1:, :]
        temp_buffer["rotation_next_obs"] = self.buffer["rotation_obs"][:, 1:, :]
        temp_buffer["angular_velocity_error_next_obs"] = self.buffer["angular_velocity_error_obs"][:, 1:, :]
        temp_buffer["action_next_obs"] = self.buffer["action_obs"][:, 1:, :]


        # Sample data.
        episode_idxs = np.random.randint(episode_num, size=batch_size)
        T = temp_buffer["length"][episode_idxs].reshape([-1])
        t_samples = np.array([np.random.randint(T[i], size=1) for i in range(batch_size)]).reshape([-1])

        if self.HER:
            # Choose data within the samples that the goal replacement is applied to
            her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
            future_offset = (np.random.uniform(size=batch_size) * (T - t_samples)).astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            future_ag = temp_buffer['position_obs'][episode_idxs[her_indexes], future_t, :3]  # Achieved goal

        del temp_buffer["length"]
        transitions = {key: temp_buffer[key][episode_idxs, t_samples].copy() for key in temp_buffer.keys()}

        if self.HER:
            # Replace the goal with an achieved goal, and reform 'target_obs'.
            current_position = transitions['position_obs'][her_indexes]
            current_velocity = -transitions['velocity_error_obs'][her_indexes]
            current_ang_velocity = -transitions['angular_velocity_error_obs'][her_indexes]
            error_state = self._error_state_func(current_position, future_ag, current_velocity, current_ang_velocity, history=True)
            transitions['position_error_obs'][her_indexes] = error_state

            next_position = transitions['position_next_obs'][her_indexes]
            next_velocity = -transitions['velocity_error_next_obs'][her_indexes]
            next_ang_velocity = -transitions['angular_velocity_error_next_obs'][her_indexes]
            next_error_state = self._error_state_func(next_position, future_ag, next_velocity, next_ang_velocity, history=True)
            transitions['position_error_next_obs'][her_indexes] = next_error_state

        # Re-compute rewards
        transitions['reward'] = self._reward_func(transitions).reshape([batch_size, 1])
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions

    def __len__(self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer_data)

    def get_length(self):
        return len(self.buffer_data)