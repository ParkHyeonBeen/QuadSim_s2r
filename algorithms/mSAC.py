#####################################################################################
'''
Soft Actor Critic (SAC) for Drone Navigation
Author  : Jongchan Baek, Yoonsu Jang
Date    : 2020.12.07
Contact : paekgga@postech.ac.kr
Reference
[1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
       Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
       with a Stochastic Actor" Deep Learning Symposium, NIPS 2017.
'''
import gym

#####################################################################################

from envs.sim2real_v1 import *
import numpy as np
import torch
import time
from tool.logger import *
from tool.utils import *
from tool.utils_model import *

eval_data = DataManager()

def worker(id, sac_trainer, rewards_queue, replay_buffer, model_path, args, log_queue, startTime=None):
    '''
    the function for sampling with multi-processing
    '''


    setup_worker_logging(rank=0, log_queue=log_queue)

    # Configure environments to train
    DETERMINISTIC = not args.train\
        if sac_trainer.worker_step.tolist()[0] < args.model_train_start_step else True

    env = Sim2RealEnv(args=args)

    # training loop
    eps = 0
    eval_freq = args.eval_frequency
    best_score = 0.
    best_error = 100.

    while sac_trainer.worker_step < args.max_interaction:

        if args.model_train_start_step <= sac_trainer.worker_step < args.model_train_start_step + args.episode_length:
            env.random_ratio = int(0)
            load_model(sac_trainer.policy_net, model_path["policy"], "policy_best")

        # Episode start
        episode_reward = 0
        state = env.reset()
        step = 0
        episode_batch = {}
        for step in range(args.episode_length):
            if sac_trainer.worker_step > args.random_action * args.num_worker:
                network_state = np.concatenate([state["position_error_obs"],
                                                state["velocity_error_obs"],
                                                state["rotation_obs"],
                                                state["angular_velocity_error_obs"]])
                action = sac_trainer.policy_net.get_action(network_state, deterministic=DETERMINISTIC)
            else:
                action = sac_trainer.policy_net.random_action()
            next_state, reward, done, success, _ = env.step(action)

            if step == 0:
                for key in state.keys():
                    episode_batch[key] = [state[key]]
                episode_batch["action"] = [action]
                episode_batch["reward"] = [np.array([reward])]
                episode_batch["done"] = [np.array([done])]
            else:
                for key in state.keys():
                    episode_batch[key] = np.vstack([episode_batch[key], [state[key]]])
                episode_batch["action"] = np.vstack([episode_batch["action"], [action]])
                episode_batch["reward"] = np.vstack([episode_batch["reward"], [reward]])
                episode_batch["done"] = np.vstack([episode_batch["done"], np.array([done])])

            state = next_state
            episode_reward += reward

            # Update networks per step
            if sac_trainer.worker_step > args.random_action * args.num_worker:
                if id < args.num_update_worker:
                    for i in range(args.update_iter):
                        try:
                            _ = sac_trainer.update(args, sac_trainer.worker_step.tolist()[0], target_entropy=-1.*env.action_dim)
                            sac_trainer.update_step += torch.tensor([1])
                        except:
                            # logging.error(traceback.format_exc())
                            pass
            if done or success:
                break

        if step < 3:
            continue

        # Save experience data
        for key in state.keys():
            episode_batch[key] = np.vstack([episode_batch[key], [state[key]]])
        replay_buffer.add(episode_batch)
        eps += 1
        sac_trainer.worker_step += torch.tensor([step + 1])
        sac_trainer.eps += torch.tensor([1])

        s = int(time.time() - startTime)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        logging.info(
            'Global Episode: {0} | Episode Reward: {1:.2f} | Local step: {7} | Global worker step: {2} | Update step: {8} | Worker id: {3} | Elapsed time: {4:02d}:{5:02d}:{6:02d}'.format(
                sac_trainer.eps.tolist()[0], episode_reward, sac_trainer.worker_step.tolist()[0], id, h, m,
                s, step + 1, sac_trainer.update_step.tolist()[0]))

        # if len(rewards) == 0: rewards.append(episode_reward)
        # else: rewards.append(rewards[-1]*0.9+episode_reward*0.1)

        # while sac_trainer.worker_step * 0.5 > sac_trainer.update_step:
        #     time.sleep(1)
        #     print("waiting for update")

        # Evaluation while training
        if id == 0 and sac_trainer.worker_step.tolist()[0] > eval_freq:
            print('--------------Evaluation start--------------')
            try:
                eval_freq += args.eval_frequency
                episode_rewards = []
                episodes_model_error = []
                success_cnt = 0
                for eval_step in range(args.num_eval):
                    episode_reward = 0
                    episode_model_error = []
                    state = env.reset()
                    for step in range(args.episode_length):
                        network_state = np.concatenate([state["position_error_obs"],
                                                        state["velocity_error_obs"],
                                                        state["rotation_obs"],
                                                        state["angular_velocity_error_obs"]])
                        action = sac_trainer.policy_net.get_action(network_state, deterministic=args.train)
                        next_state, reward, done, success, _ = env.step(action)

                        if args.develop_mode == "imn" and sac_trainer.worker_step.tolist()[0] > args.model_train_start_step:
                            sac_trainer.inv_model_net.evaluates()

                            network_states = get_model_net_input(env, state, next_state=next_state, ver=args.develop_version)
                            if args.develop_version == 1:
                                network_state, prev_network_action, next_network_state = network_states
                            else:
                                _, prev_network_action, next_network_state = network_states

                            action_hat = sac_trainer.inv_model_net(network_state, prev_network_action, next_network_state).detach().cpu().numpy()
                            episode_model_error.append(np.sqrt(np.mean((action_hat - action)**2)))

                        # env.render()
                        state = next_state
                        episode_reward += reward

                        if done or success:
                            break

                    if episode_reward > 300:
                        success_cnt += 1
                    episode_rewards.append(episode_reward)
                    episodes_model_error.append(episode_model_error)
                avg_reward = np.mean(episode_rewards)
                best_score_tmp = save_policy(sac_trainer.policy_net, best_score, avg_reward, success_cnt, model_path['policy'])
                if best_score_tmp is not None:
                    best_score = best_score_tmp

                if args.develop_mode == "imn" and sac_trainer.worker_step.tolist()[0] > args.model_train_start_step:
                    eval_error = np.mean([np.mean(episode_errors) for episode_errors in episodes_model_error], keepdims=True)
                    eval_data.put_data(eval_error)
                    best_error_tmp = save_model(sac_trainer.inv_model_net, best_error, eval_error[0],
                                                model_path[args.net_type])
                    if best_error_tmp is not None:
                        best_error = best_error_tmp
                    eval_data.plot_fig(model_path['train'] + "/model_error.png")
                    # if len(eval_data.data) > 1:
                    #     plot_variance_fig(np.mean(episodes_model_error, axis=0), np.std(episodes_model_error, axis=0),
                    #                       model_path['train'] + "/episode_model_error.png")

                rewards = [avg_reward, sac_trainer.worker_step.tolist()[0]]
                rewards_queue.put(rewards)
                logging.error(
                    'Episode Reward: {1:.2f} | Success cnt: {4} | Local Step: {3} | Global Episode: {0} | Global Worker Step: {2}'
                    .format(sac_trainer.eps.tolist()[0], avg_reward, sac_trainer.worker_step.tolist()[0],
                            step + 1, success_cnt))
            except:
                logging.error(traceback.format_exc())

    # sac_trainer.save_model(model_path)
    rewards_queue.put(None)