import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import random

from ..model_trainer.model_network import *

# from environment.models import Generator
# import torchvision.transforms as transforms

class SAC_Trainer():
    def __init__(self, env, replay_buffer, args, action_range):
        self.replay_buffer = replay_buffer
        self.transition_model_type = args.transition_type
        self.discount = args.gamma

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        action_dim = env.action_dim
        state_dim = env.state_dim * env.n_history
        hidden_dim = args.hidden_dim

        self.worker_step = torch.zeros(1, dtype=torch.int32, requires_grad=False, device='cpu')
        self.update_step = torch.zeros(1, dtype=torch.int32, requires_grad=False, device='cpu')
        self.eps = torch.zeros(1, dtype=torch.int32, requires_grad=False, device='cpu')
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)


        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = args.val_lr
        policy_lr = args.pol_lr
        alpha_lr = args.alpha_lr

        self.soft_q_optimizer = optim.Adam(list(self.soft_q_net1.parameters()) + list(self.soft_q_net2.parameters()), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.train()
        self.target_soft_q_net1.train()
        self.target_soft_q_net2.train()

        ## For training model network
        if args.develop_mode == "mrrl":
            self.model_net = ModelNetwork(state_dim, action_dim, hidden_dim, args).to(device)
            self.pid_net = PidNetwork(state_dim, action_dim, hidden_dim, args).to(device)
            self.mn_optimizer = optim.Adam(
                list(self.model_net.parameters()) + list(self.pid_net.parameters()), lr=args.model_lr)
            self.model_net.trains()
            self.pid_net.trains()

        if args.develop_mode == "imn":
            self.inv_model_net = InverseModelNetwork(state_dim, action_dim, hidden_dim, args).to(device)
            self.imn_optimizer = optim.Adam(self.inv_model_net.parameters(), lr=args.inv_model_lr)
            self.inv_model_net.trains()
            self.imn_criterion = nn.MSELoss()
            self.action = None

    def train(self, training=True):
        self.training = training
        self.policy_net.train(training)
        self.soft_q_net1.train(training)
        self.soft_q_net2.train(training)

    def eval(self):
        self.policy_net.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, args, worker_step, target_entropy=-2):

        batch_size = args.batch_size
        reward_scale = args.reward_scale
        auto_entropy = args.AUTO_ENTROPY
        gamma = args.gamma
        soft_tau = args.soft_tau
        transition = self.replay_buffer.get_batch(batch_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        network_state = np.concatenate([transition["position_error_obs"],
                                        transition["velocity_error_obs"],
                                        transition["rotation_obs"],
                                        transition["angular_velocity_error_obs"]], axis=1)
        next_network_state = np.concatenate([transition["position_error_next_obs"],
                                            transition["velocity_error_next_obs"],
                                            transition["rotation_next_obs"],
                                            transition["angular_velocity_error_next_obs"]], axis=1)
        reward = transition["reward"]
        done = transition["done"]
        action = transition["action"]

        network_state = torch.FloatTensor(network_state).to(device)
        next_network_state = torch.FloatTensor(next_network_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(np.float32(done)).to(device)

        predicted_q_value1 = self.soft_q_net1(network_state, action)
        predicted_q_value2 = self.soft_q_net2(network_state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(network_state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_network_state)
        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_network_state, new_next_action),
                                 self.target_soft_q_net2(next_network_state, new_next_action)) - self.alpha.detach() * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        q_loss = 0.5*q_value_loss1 + 0.5*q_value_loss2
        self.soft_q_optimizer.zero_grad()
        q_loss.backward()
        self.soft_q_optimizer.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(network_state, new_action, detach_encoder=True),
                                          self.soft_q_net2(network_state, new_action, detach_encoder=True))
        policy_loss = (self.alpha.detach() * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Training model network
        if args.develop_mode == "imn" and worker_step > args.max_interaction/100:
            self.inv_model_net.trains()
            action_hat = self.inv_model_net(network_state, next_network_state)
            if self.action is not None:
                action_hat = self.action + 0.5 * (action_hat - self.action)
            model_loss = F.smooth_l1_loss(action, action_hat).mean()
            self.imn_optimizer.zero_grad()
            model_loss.backward()
            self.imn_optimizer.step()
            self.action = action_hat

        # Training alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = (-self.alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return predicted_new_q_value.mean()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, encoder_feature_dim=50, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

        # torch.nn.init.xavier_uniform(self.linear1.weight)
        # torch.nn.init.xavier_uniform(self.linear2.weight)
        # torch.nn.init.xavier_uniform(self.linear3.weight)

        torch.nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity='relu')


    def forward(self, state, detach_encoder=False):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, encoder_feature_dim=50, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs+num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

        # torch.nn.init.xavier_uniform(self.linear1.weight)
        # torch.nn.init.xavier_uniform(self.linear2.weight)
        # torch.nn.init.xavier_uniform(self.linear3.weight)

        torch.nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity='relu')

    def forward(self, state, action, detach_encoder=False):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., encoder_feature_dim=50, init_w=3e-3,
                 log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.zero_()

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        # self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.zero_()

        # torch.nn.init.xavier_uniform(self.linear1.weight)
        # torch.nn.init.xavier_uniform(self.linear2.weight)
        # torch.nn.init.xavier_uniform(self.mean_linear.weight)
        # torch.nn.init.xavier_uniform(self.log_std_linear.weight)

        torch.nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.mean_linear.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.log_std_linear.weight, nonlinearity='relu')

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state, detach_encoder=False):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mean, log_std = self.forward(state, detach_encoder=True)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action

    def random_action(self, ):
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range * a.numpy()
