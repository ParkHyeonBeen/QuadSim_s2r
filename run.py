
import argparse

from algorithms.source.SAC_trainer import *
from algorithms.source.replay_buffer import *
from algorithms.mSAC import *
from envs.sim2real_v1 import *
from tool.utils import *

import torch
torch.multiprocessing.set_start_method('spawn', force=True) # critical for make multiprocessing work
# fork, forkserver for Linux
# spawn for Windows 10
import torch.multiprocessing as mp
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

# Argument
parser = argparse.ArgumentParser(description="QuadSim_mSAC")

# Base
parser.add_argument("--base-path", default="/home/phb/ETRI/QuadSim_s2r/", type=str, help="base path of the current project")
parser.add_argument("--gpu", default=True, type=bool, help="If use gpu, True")
parser.add_argument("--device-idx", default=0, type=int, help="a index about gpu device")

# Main controller
parser.add_argument("--train", default=False, type=bool, help="If True, run_train")
parser.add_argument("--develop-mode", "-dm", default='imn', type=str,
                    help="none   : only policy network,"
                         "mrrl   : model reference rl,"
                         "mn_mrrl: model reference rl with model network,"
                         "imn    : inverse model network")
parser.add_argument("--env-name", default='QuadRotor-v0', type=str, help="If True, run_train")

# For test
parser.add_argument("--test_eps", default=50, type=int, help="The number of test episode using trained policy.")
parser.add_argument("--result_name", default="0730_HD128_IA120_RR0.1_BS_7.6e7", type=str, help="Checkpoint path to a pre-trained model.")

# For train
# ModelNet
parser.add_argument("--model_lr", default=3e-4, type=float, help="Learning rate for model network update.")
parser.add_argument("--inv_model_lr", default=3e-4, type=float, help="Learning rate for inverse model network update.")

# SAC
parser.add_argument("--name", default="mSAC", type=str, help="Trained model is saved with this name.")
parser.add_argument("--num_worker", default=6, type=int, help="The number of agents for collect data.")
parser.add_argument("--num_update_worker", default=2, type=int, help="The number of agents for update networks.")
parser.add_argument("--max_interaction", default=4e6, type=int, help="Maximum interactions for training.")
parser.add_argument("--episode_length", default=1000, type=int, help="Maximum steps in an episode.")
parser.add_argument("--random_action", default=1000, type=int, help="The number of random actions to be executed at the start of training.")
parser.add_argument("--sampling_time", default=0.01, type=float, help="Sampling time for policy.")
parser.add_argument("--n_history", default=3, type=int, help="state history stack")

parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--soft_tau", default=0.01, type=float, help="Averaging for target network.")
parser.add_argument("--reward_scale", default=1, type=float, help="Reward scale in SAC.")
parser.add_argument("--dense_reward", default=True, type=bool, help="The use of dense reward. If False, sparse reward is used.")
parser.add_argument("--AUTO_ENTROPY", default=True, type=bool, help="Training entrophy.")
parser.add_argument("--hidden_dim", default=128, type=int, help="Network hidden state dimension.")

parser.add_argument("--eval_frequency", default=5000, type=int, help="Evaluation frequency in the aspect of the number of agent local steps.")
parser.add_argument("--num_eval", default=10, type=int, help="The number of evaluations at once.")

parser.add_argument("--buffer_size", default=3e6, type=int, help="Buffer size.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--future_p", default=0.7, type=float, help="The probability of replacing goals for a sample.")

parser.add_argument("--random_ratio", default=0.2, type=float, help="Random ratio for quadrotor parameter.")
parser.add_argument("--update_iter", default=1, type=int, help="Update iteration for one step.")
parser.add_argument("--pol_lr", default=3e-4, type=float, help="Learning rate for actor (policy) update.")
parser.add_argument("--val_lr", default=3e-4, type=float, help="Learning rate for critic (Q, V) update.")
parser.add_argument("--alpha_lr", default=3e-4, type=float, help="Learning rate for alpha (a) update.")
parser.add_argument("--target_vf_alpha", default=0.995, type=float, help="Exponential moving average coefficient for updating target value function.")

parser.add_argument("--transition_type", default="probabilistic", type=str, help="Transition model type: deterministic, probabilistic, ensemble")
parser.add_argument("--bisim_coef", default=0.5, type=float, help="Bisimulation coefficient.")
parser.add_argument("--bisim", default=False, type=bool, help="If True, Bisimulation encoder training is applied.")
parser.add_argument("--restore", default=False, type=bool, help="If True, pre-trained model is restored.")
parser.add_argument("--HER", default=False, type=bool, help="If True, replay buffer is applied HER.")

# for environment
parser.add_argument("--init_max_pbox", default=3., type=float, help="max initial position near goal")
parser.add_argument("--init_max_ang", default=45, type=float, help="max initial degree angle for roll and pitch")
parser.add_argument("--init_max_vel", default=1., type=float, help="max initial velocity")
parser.add_argument("--init_max_ang_vel", default=2.*np.pi, type=float, help="max initial angular velocity")
parser.add_argument("--thrust_noise_sigma", default=0.05, type=float, help="motor noise scale")
parser.add_argument("--step_size", default=0.005, type=float, help="RK4 step size")
parser.add_argument("--quad_ver", default="v2", type=str, choices=["v1", "v2"], help="Quadrotor version, v1: 3d printed micro drone, v2: carbon mini drone")
parser.add_argument("--lag_ratio", default=0.15, type=float, help="motor lag ratio for diff factor")
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

    log_dir = create_log_directories("./results", args.env_name)
    update_log_file = log_dir['train'] + "/step.log"
    config_log_file = log_dir['base'] + "/config.log"
    evaluation_log_file = log_dir['train'] + "/evaluation.log"
    config_logger = create_config_logger(args, file=config_log_file)
    startTime = time.time()

    log_queue = setup_primary_logging(update_log_file, evaluation_log_file)

    # Training
    if args.train:

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

        rewards_queue = mp.Queue()  # used for get rewards from all processes and plot the curve

        processes = []
        rewards = []
        eval_step = []
        eval_num = 1

        for i in range(args.num_worker):
            process = Process(target=worker, args=(
            i, sac_trainer, rewards_queue, replay_buffer, log_dir, args, log_queue, startTime))  # the args contain shared and not shared
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
                plot_fig(rewards, log_dir['train'])

        [p.join() for p in processes]  # finished at the same time

    # Test trained policy
    else:
        np.random.seed(77)
        load_model(sac_trainer.policy_net, log_dir["policy"], "policy_best")
        load_model(sac_trainer.inv_model_net, log_dir["policy"], "policy_best")
        env = Sim2RealEnv(sample_time=args.sampling_time, args=args)

        success_rate = 0
        avg_reward = 0
        suc_reward = 0

        for eps in range(args.test_eps):
            state = env.reset()
            episode_reward = 0

            for step in range(args.episode_length):
                network_state = np.concatenate([state["position_error_obs"],
                                                state["velocity_error_obs"],
                                                state["rotation_obs"],
                                                state["angular_velocity_error_obs"]])
                action = sac_trainer.policy_net.get_action(network_state, deterministic=not args.train)
                next_state, reward, done, success = env.step(action)
                # env.render()
                # time.sleep(0.005)
                episode_reward += reward
                state = next_state
                if done or success:
                    break

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            avg_reward += episode_reward
            if episode_reward > 300:
                suc_reward += episode_reward
                success_rate += 1.
        suc_reward /= success_rate
        success_rate /= args.test_eps
        avg_reward /= args.test_eps
        print('Success rate: ', success_rate*100, '| Average Reward: ', avg_reward, '| Success Reward: ',suc_reward)