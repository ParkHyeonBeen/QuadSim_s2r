import argparse

import torch
from copy import deepcopy

from algorithms.mSAC import *
from algorithms.source.SAC_trainer import *
from tool.utils import *

torch.multiprocessing.set_start_method('spawn', force=True)  # critical for make multiprocessing work
# fork, forkserver for Linux
# spawn for Windows 10
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing.managers import BaseManager

# Argument
parser = argparse.ArgumentParser(description="QuadSim_mSAC")

# Base
parser.add_argument("--base_path", default="/home/phb/ETRI/QuadSim_s2r/", type=str,
                    help="base path of the current project")
parser.add_argument("--gpu", default=True, type=bool, help="If use gpu, True")
parser.add_argument("--device_idx", default=0, type=int, help="a index about gpu device")

# Main controller
parser.add_argument("--train", default="True", type=str2bool, help="If True, run_train")
parser.add_argument("--develop_mode", "-dm", default='imn', type=str,
                    help="none   : only policy network,"
                         "imn    : inverse model network")
parser.add_argument("--env_name", default='QuadRotor-v0', type=str, help="If True, run_train")
parser.add_argument("--net_type", default='dnn', type=str, help="dnn, bnn, rbf")
parser.add_argument("--develop_version", '-dv', default=0, type=int,
                    help="0 : network state is same with policy network state"
                         "others :  network state is different with policy network state")

# For test
parser.add_argument("--test_eps", '-te', default=100, type=int, help="The number of test episode using trained policy.")
parser.add_argument("--render", default="False", type=str2bool)
parser.add_argument("--result_name", "-rn", default="1001-2208QuadRotor-v0", type=str,
                    help="Checkpoint path to a pre-trained model.")
parser.add_argument('--use_prev_policy', '-upp', default='False', type=str2bool, help="if True, use prev best policy")
parser.add_argument("--model_on", default="True", type=str2bool, help="if True, activate model network")
parser.add_argument("--result_ver", default="best", type=str, help="choose the version of the results")
parser.add_argument("--set_goal", default=[0., 0., 2.], help="set goal")
parser.add_argument("--set_path", default="none", type=str, help="none, circle, sinewave")

## To make a uncertain system
parser.add_argument('--num_case', '-nc', default=20, type=int, help='the number of cases in certain range')
parser.add_argument('--which_kind', '-wk', default='disturb', type=str, help='disturb, uncertain, noise')

parser.add_argument('--max_disturb', '-xd', default=0.2, type=float, help='max mag of disturbance for action')
parser.add_argument('--min_disturb', '-nd', default=0.0, type=float, help='min mag of disturbance for action')
parser.add_argument('--max_uncertain', '-xu', default=0.8, type=float, help='max mag of uncertainty for model param')
parser.add_argument('--min_uncertain', '-nu', default=0.0, type=float, help='min mag of uncertainty for model param')
parser.add_argument('--max_noise', '-xn', default=0.1, type=float, help='max std of gaussian noise for state')
parser.add_argument('--min_noise', '-nn', default=0.0, type=float, help='min std of gaussian noise for state')

# For train
# ModelNet
parser.add_argument("--model_lr", default=3e-4, type=float, help="Learning rate for model network update.")
parser.add_argument("--inv_model_lr", default=3e-4, type=float, help="Learning rate for inverse model network update.")
parser.add_argument('--model-kl-weight', default=0.00001, type=float)
parser.add_argument('--inv-model-kl-weight', default=0.00001, type=float)
parser.add_argument('--model_train_start_step', default=1.5e7, type=int)
parser.add_argument('--reg_weight', default=1.0e-6, type=int)

# SAC
parser.add_argument("--name", default="mSAC", type=str, help="Trained model is saved with this name.")
parser.add_argument("--num_worker", default=10, type=int, help="The number of agents for collect data.")
parser.add_argument("--num_update_worker", default=3, type=int, help="The number of agents for update networks.")
parser.add_argument("--max_interaction", default=6e8, type=int, help="Maximum interactions for training.")
parser.add_argument("--episode_length", default=1000, type=int, help="Maximum steps in an episode.")
parser.add_argument("--random_action", default=1000, type=int,
                    help="The number of random actions to be executed at the start of training.")
parser.add_argument("--sampling_time", default=0.01, type=float, help="Sampling time for policy.")
parser.add_argument("--n_history", default=3, type=int, help="state history stack")

parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--soft_tau", default=0.01, type=float, help="Averaging for target network.")
parser.add_argument("--reward_scale", default=1, type=float, help="Reward scale in SAC.")
parser.add_argument("--dense_reward", default=True, type=bool,
                    help="The use of dense reward. If False, sparse reward is used.")
parser.add_argument("--AUTO_ENTROPY", default=True, type=bool, help="Training entrophy.")
parser.add_argument("--hidden_dim", default=128, type=int, help="Network hidden state dimension.")

parser.add_argument("--eval_frequency", default=100000, type=int,
                    help="Evaluation frequency in the aspect of the number of agent local steps.")
parser.add_argument("--num_eval", default=10, type=int, help="The number of evaluations at once.")

parser.add_argument("--buffer_size", default=3e6, type=int, help="Buffer size.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--future_p", default=0.7, type=float, help="The probability of replacing goals for a sample.")

parser.add_argument("--random_ratio", '-rr', default=0.05, type=float, help="Random ratio for quadrotor parameter.")
parser.add_argument("--update_iter", default=1, type=int, help="Update iteration for one step.")
parser.add_argument("--pol_lr", default=3e-4, type=float, help="Learning rate for actor (policy) update.")
parser.add_argument("--val_lr", default=3e-4, type=float, help="Learning rate for critic (Q, V) update.")
parser.add_argument("--alpha_lr", default=3e-4, type=float, help="Learning rate for alpha (a) update.")
parser.add_argument("--target_vf_alpha", default=0.995, type=float,
                    help="Exponential moving average coefficient for updating target value function.")

parser.add_argument("--transition_type", default="probabilistic", type=str,
                    help="Transition model type: deterministic, probabilistic, ensemble")
parser.add_argument("--bisim_coef", default=0.5, type=float, help="Bisimulation coefficient.")
parser.add_argument("--bisim", default=False, type=bool, help="If True, Bisimulation encoder training is applied.")
parser.add_argument("--restore", default=False, type=bool, help="If True, pre-trained model is restored.")
parser.add_argument("--HER", default=False, type=bool, help="If True, replay buffer is applied HER.")

# for environment
parser.add_argument("--init_max_pbox", '-ip', default=3., type=float, help="max initial position near goal")
parser.add_argument("--init_max_ang", '-ia', default=45, type=float, help="max initial degree angle for roll and pitch")
parser.add_argument("--init_max_vel", '-iv', default=0.5, type=float, help="max initial velocity")
parser.add_argument("--init_max_ang_vel", '-iav', default=1. * np.pi, type=float, help="max initial angular velocity")
parser.add_argument("--thrust_noise_sigma", default=0.05, type=float, help="motor noise scale")
parser.add_argument("--step_size", default=0.005, type=float, help="RK4 step size")
parser.add_argument("--lambda_t", default=1e-1, type=float, help="Temporal smoothness for policy loss")
parser.add_argument("--lambda_s", default=5e-1, type=float, help="Spatial smoothness for policy loss")
parser.add_argument("--eps_p", default=1e-1, type=float, help="Standard deviation for spatial smoothness")
parser.add_argument("--quad_ver", default="v2", type=str, choices=["v1", "v2"],
                    help="Quadrotor version, v1: 3d printed micro drone, v2: carbon mini drone")
parser.add_argument("--lag_ratio", default=0.5, type=float, help="motor lag ratio for diff factor")
parser.add_argument("--gravity", default=9.8066, type=float, help="")
parser.add_argument("--tc", default=0.060, type=float, help="time constant between motor and propeller")
parser.add_argument("--alpha", default=2.5, type=float, help="disturbance")
parser.add_argument("--delta", default=0.24, type=float, help="disturbance")
parser.add_argument("--sigma", default=1000., type=float, help="disturbance")

args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda:" + str(args.device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

if __name__ == '__main__':
    # env initialize
    env = Sim2RealEnv(args=args)

    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(args, env)  # share the replay buffer through manager

    # hyper-parameters for RL training, no need for sharing across processes
    sac_trainer = SAC_Trainer(env, replay_buffer, args=args, action_range=1.)

    # Training
    if args.train:
        log_dir = create_log_directories("./results", args.env_name)
        update_log_file = log_dir['train'] + "/step.log"
        config_log_file = log_dir['base'] + "/config.log"
        evaluation_log_file = log_dir['train'] + "/evaluation.log"
        config_logger = create_config_logger(args, file=config_log_file)
        startTime = time.time()

        log_queue = setup_primary_logging(update_log_file, evaluation_log_file)

        # share the global parameters in multiprocessing
        sac_trainer.soft_q_net1.share_memory()
        sac_trainer.soft_q_net2.share_memory()
        sac_trainer.target_soft_q_net1.share_memory()
        sac_trainer.target_soft_q_net2.share_memory()
        sac_trainer.policy_net.share_memory()
        sac_trainer.log_alpha.share_memory_()  # variable
        sac_trainer.worker_step.share_memory_()
        sac_trainer.update_step.share_memory_()
        sac_trainer.eps.share_memory_()
        share_parameters(sac_trainer.soft_q_optimizer)
        share_parameters(sac_trainer.policy_optimizer)
        share_parameters(sac_trainer.alpha_optimizer)

        if args.develop_mode == "imn":
            sac_trainer.inv_model_net.share_memory()
            share_parameters(sac_trainer.imn_optimizer)

        rewards_queue = mp.Queue()  # used for get rewards from all processes and plot the curve

        processes = []
        rewards = []
        eval_step = []
        eval_num = 1

        for i in range(args.num_worker):
            process = Process(target=worker, args=(
                i, sac_trainer, rewards_queue, replay_buffer, log_dir, args, log_queue,
                startTime))  # the args contain shared and not shared
            process.daemon = True  # all processes closed when the main stops
            processes.append(process)

        for p in processes:
            p.start()
            time.sleep(5)

        while True:  # keep geting the episode reward from the queue
            r = rewards_queue.get()
            if r is not None:
                rewards.append(r[0])
                eval_step.append(r[1])
                del r
            else:
                break
            if len(rewards) > 0:
                plot_fig(rewards, log_dir['train'] + "/reward.png")

        [p.join() for p in processes]  # finished at the same time

    # Test trained policy
    else:
        np.random.seed(77)

        eval_reward = DataManager()
        eval_success = DataManager()
        if args.model_on:
            eval_model_error = DataManager()

        log_dir = load_log_directories(args.result_name)
        load_model(sac_trainer.policy_net, log_dir["policy"], "policy_best")
        result_txt = open(log_dir["test"] + "/test_result_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind + ".txt", 'w')
        env = Sim2RealEnv(args=args)

        if args.model_on:
            if args.net_type == "dnn":
                load_model(sac_trainer.inv_model_net, log_dir[args.net_type], "better_" + args.net_type)
            else:
                load_model(sac_trainer.inv_model_net, log_dir[args.net_type], args.result_ver + "_" + args.net_type)
            sac_trainer.inv_model_net.evaluates()

        if args.which_kind == "disturb":
            min_case = args.min_disturb
            max_case = args.max_disturb
        elif args.which_kind == "uncertain":
            min_case = args.min_uncertain
            max_case = args.max_uncertain
        else:
            min_case = args.min_noise
            max_case = args.max_noise

        normal_mean = 0
        normal_std = 0

        for case in np.round(np.linspace(min_case, max_case, args.num_case + 1), 3):
            if args.which_kind == "disturb":
                env.dist_scale = case
                print("disturbance scale: ", case * 100, " percent of max thrust", file=result_txt)
                print("disturbance scale: ", case * 100, " percent of max thrust")

                eval_reward.get_xticks(np.round(case * 100, 3))
                eval_success.get_xticks(np.round(case * 100, 3))
                if args.model_on:
                    eval_model_error.get_xticks(np.round(case * 100, 3))

            elif args.which_kind == "uncertain":
                env.random_ratio = (case - (max_case / 2)) * 2
                print("uncertainty scale: ", case * 100, " percent of init property", file=result_txt)
                print("uncertainty scale: ", case * 100, " percent of init property")

                eval_reward.get_xticks(np.round(case * 100, 3))
                eval_success.get_xticks(np.round(case * 100, 3))
                if args.model_on:
                    eval_model_error.get_xticks(np.round(case * 100, 3))

            else:
                print("standard deviation of state noise: ", case, file=result_txt)
                print("standard deviation of state noise: ", case)

                eval_reward.get_xticks(np.round(case, 3))
                eval_success.get_xticks(np.round(case, 3))
                if args.model_on:
                    eval_model_error.get_xticks(np.round(case, 3))

            reward_list = []
            model_error_list = []
            success_rate = 0
            suc_reward = 0

            for eps in range(args.test_eps):
                state = env.reset()
                episode_reward = 0
                # p = state["position_error_obs"]
                # v = state["velocity_error_obs"]
                # r = state["rotation_obs"]
                # w = state["angular_velocity_error_obs"]
                # a = state["action_obs"]
                # pos = p[:3]
                # vel = -v[:3]
                # rpy = np.array([math.atan2(r[0], r[1]), math.atan2(r[2], r[3]), math.atan2(r[4], r[5])])
                # angvel = -w[:3]
                # policy = a[:4]
                # force = np.zeros(4)
                step = 0
                episode_model_error = []
                dist = np.zeros(env.action_dim)

                for step in range(args.episode_length):
                    action = get_action(sac_trainer.policy_net, state)

                    if args.model_on:
                        action_dob = (action - dist) if args.which_kind != "noise" else action

                        next_state, reward, done, success, _ = env.step(action_dob)

                        next_state_imn = deepcopy(next_state)
                        if args.which_kind == "noise":
                            for k in next_state_imn.keys():
                                next_state_imn[k] = np.random.normal(next_state_imn[k], case)

                        network_state, prev_network_action, next_network_state = \
                            get_model_net_input(env, state, next_state=next_state_imn, ver=args.develop_version)

                        action_hat = sac_trainer.inv_model_net(network_state, prev_network_action,
                                                               next_network_state).detach().cpu().numpy()[0]
                        dist = action_hat - action
                        dist = np.clip(dist, -1.0, 1.0)
                        episode_model_error.append(np.sqrt(np.mean(dist ** 2)))
                    else:
                        next_state, reward, done, success, f = env.step(action)

                    episode_reward += reward
                    state = next_state
                    if args.render:
                        env.render()

                    # p = state["position_error_obs"]
                    # v = state["velocity_error_obs"]
                    # r = state["rotation_obs"]
                    # w = state["angular_velocity_error_obs"]
                    # a = state["action_obs"]
                    # pos = np.vstack((pos, p[:3]))
                    # vel = np.vstack((vel, -v[:3]))
                    # rpy = np.vstack(
                    #     (rpy, np.array([math.atan2(r[0], r[1]), math.atan2(r[2], r[3]), math.atan2(r[4], r[5])])))
                    # angvel = np.vstack((angvel, -w[:3]))
                    # policy = np.vstack((policy, a[:4]))
                    # force = np.vstack((force, f))

                    if done or success:
                        break

                # when you want to know specific data
                # eval_plot(step, pos, vel, rpy, angvel, policy, force)

                print('Episode: ', eps, '| Episode Reward: ', episode_reward,
                      '| Mean and Std of Episode Model error: ',
                      np.mean(episode_model_error), np.std(episode_model_error))

                reward_list.append(episode_reward)
                model_error_list.append(np.mean(episode_model_error))

                if episode_reward > 300:
                    suc_reward += episode_reward
                    success_rate += 1.

            if success_rate != 0:
                suc_reward /= success_rate
            else:
                suc_reward = 0.

            success_rate /= args.test_eps
            avg_reward = sum(reward_list) / args.test_eps
            eval_reward.put_data((np.mean(reward_list), np.std(reward_list)))

            if args.model_on:
                if case == 0:
                    normal_mean = np.mean(model_error_list)
                    normal_std = np.std(model_error_list)

                eval_model_error.put_data((np.mean(model_error_list) - normal_mean, np.std(model_error_list) - normal_std))
            eval_success.put_data(success_rate * 100)

            print('Success rate: ', success_rate * 100, '| Average Reward: ', avg_reward, '| Success Reward: ',
                  suc_reward, file=result_txt)
            print('Success rate: ', success_rate * 100, '| Average Reward: ', avg_reward, '| Success Reward: ',
                  suc_reward)

        eval_reward.plot_variance_fig(log_dir["test"] + "/reward_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, need_xticks=True)
        eval_success.bar_fig(log_dir["test"] + "/success_rate_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind)
        if args.model_on:
            eval_model_error.plot_variance_fig(log_dir["test"] + "/model_error_%s" % time.strftime(
                "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, need_xticks=True)

        eval_reward.save_data(log_dir["test"], "/reward_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, numpy=True)
        eval_reward.save_data(log_dir["test"], "/reward_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind)
        if args.model_on:
            eval_model_error.save_data(log_dir["test"], "/model_error_%s" % time.strftime(
                "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, numpy=True)
            eval_model_error.save_data(log_dir["test"], "/model_error_%s" % time.strftime(
                "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind)
        eval_success.save_data(log_dir["test"], "/success_rate_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind, numpy=True)
        eval_success.save_data(log_dir["test"], "/success_rate_%s" % time.strftime(
            "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.which_kind)
        result_txt.close()
