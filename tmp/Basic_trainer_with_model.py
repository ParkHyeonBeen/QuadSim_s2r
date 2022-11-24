import numpy as np

from Common.Utils_model import *
from Network.Model_Network import *

class Model_trainer():
    def __init__(self, env, test_env, algorithm,
                 state_dim, action_dim,
                 max_action, min_action,
                 args, args_tester=None):

        self.args = args
        self.args_tester = args_tester

        self.n_history = self.args.n_history
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.domain_type = self.args.domain_type
        self.env_name = self.args.env_name
        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm

        self.max_action = max_action
        self.min_action = min_action

        self.discrete = self.args.discrete
        self.max_step = self.args.max_step

        self.eval = self.args.eval
        self.eval_episode = self.args.eval_episode
        self.eval_step = self.args.eval_step

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.test_num = 0

        # score
        self.score = None
        self.loss = None

        self.log = self.args.log
        self.frame_skip = self.test_env.env.frame_skip

        if self.args_tester is None:
            self.render = self.args.render
            self.path = self.args.path
        else:
            self.render = self.args_tester.render
            self.path = self.args_tester.path
            self.test_episode = self.args_tester.test_episode

        self.deepdob = None
        self.mrap = None
        self.eval_data = DataManager()
        self.train_data = DataManager()

        if args.net_type == 'DNN':
            self.models = create_models(
                self.state_dim, self.action_dim, self.algorithm, self.args, net_type="dnn")
        elif args.net_type == 'BNN':
            self.models = create_models(
                self.state_dim, self.action_dim, self.algorithm, self.args, net_type="bnn")
        else:
            self.models = create_models(
                self.state_dim, self.action_dim, self.algorithm, self.args, net_type="dnn,bnn")

        """
        fail score for the environments which don't have uncertain criterion about fail
        fail score = maximum score * 0.2

        Half cheetah    : 1183.44
        Ant             : 647.3822
        """
        self.fail_score = 2 * 647.3822

        # For Testing
        if self.args_tester is not None:
            load_models(args_tester, self.models)

    def evaluate(self):
        self.eval_num += 1
        episode = 0
        reward_list = []
        error_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
            if episode >= self.eval_episode:
                break
            episode += 1
            eval_reward = 0
            episode_error = []
            observation = self.test_env.reset()
            done = False

            model_error = np.zeros_like(observation)
            mem_observation = np.concatenate([observation] * self.n_history)
            mem_action = np.concatenate([np.zeros(self.action_dim)] * self.n_history)

            while not done:
                self.local_step += 1
                action = self.algorithm.eval_action(observation)
                pid_action = self.algorithm.get_pid_action(model_error, self.env.unwrapped.dt)
                action += np.squeeze(pid_action)
                action = np.clip(action, -1, 1)
                env_action = denormalize(action, self.max_action, self.min_action)

                # update the state and action history
                mem_observation[self.state_dim:] = mem_observation[:self.state_dim * (self.n_history - 1)]
                mem_observation[:self.state_dim] = observation
                mem_action[self.action_dim:] = mem_action[:self.action_dim * (self.n_history-1)]
                mem_action[:self.action_dim] = action

                next_observation, reward, done, _ = self.test_env.step(env_action)

                next_observation_hat = self.algorithm.get_next_state(mem_observation, mem_action[:self.action_dim * 2])
                model_error = next_observation_hat - next_observation
                episode_error.append(np.sqrt(np.mean(model_error**2)))

                if self.render:
                    self.test_env.render()

                eval_reward += reward
                observation = next_observation

                if self.local_step == self.env.spec.max_episode_steps:
                    alive_cnt += 1

            reward_list.append(eval_reward)
            error_list.append(np.mean(episode_error))

        self.eval_data.put_data(np.array([np.mean(error_list), np.std(error_list)]))
        score_now = sum(reward_list) / len(reward_list)
        alive_rate = alive_cnt / self.eval_episode

        eval_loss, mean, std, error_max = validate_measure(error_list)

        if self.eval_num == 1:
            self.score = score_now
            self.loss = eval_loss

        _score = save_policys(self.algorithm, self.score, score_now, alive_rate, self.path)

        if _score is not None:
            self.score = _score

        loss_new = save_models(self.loss, eval_loss, self.path, self.algorithm)
        if loss_new is not None:
            self.loss = loss_new
            # self.loss[loss_new[0]] = loss_new[1]

        if self.eval_num > 1:
            self.eval_data.plot_variance_fig(self.path + "saved_log/evaluation_error.png")

        if self.total_step == self.args.buffer_size:
            self.algorithm.buffer.save_buffer(self.path, 'by_full')
        elif self.total_step > self.args.buffer_size:
            self.algorithm.buffer.save_buffer(self.path, 'after_full')

        print("Eval  | Average Reward: {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}"
              .format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list), 100*alive_rate))

        print("Cost  | ", self.args.develop_mode, self.args.net_type, " | Average:", mean, ", Std:", std, ", Max:", error_max)
        self.test_env.close()

    def run(self):
        reward_list = []
        error_mean_list = []
        error_std_list = []
        while True:
            if self.total_step > self.max_step:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.episode_model_error = []
            self.local_step = 0

            observation = self.env.reset()
            self.test_env.reset()
            done = False

            model_error = np.zeros_like(observation)
            mem_observation = np.concatenate([observation] * self.n_history)
            mem_action = np.concatenate([np.zeros(self.action_dim)] * self.n_history)

            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.render:
                    self.env.render()

                if self.total_step <= self.algorithm.training_start:
                    env_action = self.env.action_space.sample()
                    action = normalize(env_action, self.max_action, self.min_action)
                else:
                    action = self.algorithm.get_action(observation)
                    pid_action = self.algorithm.get_pid_action(model_error, self.env.unwrapped.dt)
                    action += np.squeeze(pid_action)
                    action = np.clip(action, -1, 1)
                    env_action = denormalize(action, self.max_action, self.min_action)

                # update the state and action history
                mem_observation[self.state_dim:] = mem_observation[:self.state_dim * (self.n_history - 1)]
                mem_observation[:self.state_dim] = observation
                mem_action[self.action_dim:] = mem_action[:self.action_dim * (self.n_history-1)]
                mem_action[:self.action_dim] = action

                next_observation, reward, done, info = self.env.step(env_action)
                next_observation_hat = self.algorithm.get_next_state(mem_observation, mem_action[:self.action_dim * 2])
                model_error = next_observation_hat - next_observation
                self.episode_model_error.append(np.sqrt(np.mean(model_error**2)))

                if self.local_step + 1 == 1000:
                    real_done = 0.
                else:
                    real_done = float(done)

                self.episode_reward += reward
                self.algorithm.buffer.add(mem_observation, mem_action, reward, next_observation, real_done)

                observation = next_observation

                if self.total_step >= self.algorithm.training_start and not done:
                    loss_list = self.algorithm.train(self.algorithm.training_step)

                if self.eval is True and self.total_step % self.eval_step == 0:
                    self.evaluate()
                    if self.args.numpy is False:
                        df = pd.DataFrame(reward_list)
                        df.to_csv(self.path + "saved_log/reward" + ".csv")
                    else:
                        df = np.array(reward_list)
                        plot_fig(reward_list, self.path + "saved_log/reward.png")
                        if self.eval_num > 1:
                            plot_variance_fig(error_mean_list, error_std_list, self.path + "saved_log/train_error.png")
                            # np.save(self.path + "saved_log/reward" + ".npy", df)

            reward_list.append(self.episode_reward)
            if self.eval_num > 1:
                error_mean_list.append(np.mean(self.episode_model_error))
                error_std_list.append(np.std(self.episode_model_error))

            self.train_data.put_data(np.mean(abs(model_error)))
            print("Train | Episode: {}, Reward: {:.2f}, Local_step: {}, Total_step: {}".format(
                self.episode, self.episode_reward, self.local_step, self.total_step))
        self.env.close()

    def test(self):
        self.test_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
            alive = False
            if episode >= self.test_episode:
                break
            episode += 1
            eval_reward = 0
            loss = 0

            observation = self.test_env.reset()

            done = False

            while not done:
                self.local_step += 1

                action = self.algorithm.eval_action(observation)
                env_action = denormalize(action, self.max_action, self.min_action)

                if self.args_tester.add_noise is True and self.args_tester.noise_to == 'action':
                    env_action, _ = add_noise(env_action, scale=self.args_tester.noise_scale)
                if self.args_tester.add_disturbance is True and self.args_tester.disturbance_to == 'action':
                    env_action, _ = add_disturbance(env_action, self.local_step,
                                                    self.env.spec.max_episode_steps,
                                                    scale=self.args_tester.disturbance_scale,
                                                    frequency=self.args_tester.disturbance_frequency)
                next_observation, reward, done, _ = self.test_env.step(env_action)

                if self.args_tester.add_noise is True and self.args_tester.noise_to == 'state':
                    next_observation, _ = add_noise(next_observation, scale=self.args_tester.noise_scale)
                if self.args_tester.add_disturbance is True and self.args_tester.disturbance_to == 'state':
                    next_observation, _ = add_disturbance(next_observation, self.local_step,
                                                          self.test_env.spec.max_episode_steps,
                                                          scale=self.args_tester.disturbance_scale,
                                                          frequency=self.args_tester.disturbance_frequency)

                if self.render:
                    self.test_env.render()

                eval_reward += reward
                observation = next_observation
                if self.local_step == self.env.spec.max_episode_steps and eval_reward > self.fail_score:   # 1183.44 halfcheetah 647.3822 ant
                    alive_cnt += 1
                    alive = True

            if eval_reward < 0:
                eval_reward = 0
            print("Eval of {}th episode  | Episode Reward {:.2f}, alive : {}".format(episode, eval_reward, alive))
            reward_list.append(eval_reward)

        print(
            "Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}".format(
                sum(reward_list) / len(reward_list), max(reward_list), min(reward_list), np.std(reward_list),
                100 * (alive_cnt / self.test_episode)))
        self.test_env.close()

        return sum(reward_list) / len(reward_list), \
               max(reward_list), min(reward_list), \
               np.std(reward_list), \
               100 * (alive_cnt / self.test_episode)


