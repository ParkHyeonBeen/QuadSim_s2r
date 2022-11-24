import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbnn as bnn
from torch.optim import lr_scheduler
import numpy as np

from Common.Utils import weight_init


class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, args,
                 net_type=None, hidden_dim=256):
        super(DynamicsNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.n_history = args.n_history
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args.batch_size

        if self.net_type == "DNN":
            self.state_net = nn.Sequential(
                nn.Linear(self.state_dim*self.n_history, int(hidden_dim/2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.action_net = nn.Sequential(
                nn.Linear(self.action_dim * 2, int(hidden_dim/2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.next_state_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.15),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.state_dim)
            )

        if self.net_type == "BNN":
            self.state_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                in_features=self.state_dim * self.n_history, out_features=int(hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.action_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                in_features=self.action_dim * 2, out_features=int(hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.next_state_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=int(hidden_dim / 2), out_features=int(hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU(),
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=self.state_dim)
            )

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = args.model_kl_weight

        self.apply(weight_init)

    def forward(self, state, action):

        if type(state) is not torch.Tensor:
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.unsqueeze(0)
        if type(action) is not torch.Tensor:
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
            action = action.unsqueeze(0)

        state = self.state_net(state)
        action = self.action_net(action)

        z = torch.cat([state, action], dim=-1)
        next_state = self.next_state_net(z)

        return next_state

    def trains(self):
        self.state_net.train()
        self.action_net.train()
        self.next_state_net.train()

    def evals(self):
        self.state_net.eval()
        self.action_net.eval()
        self.next_state_net.eval()

class PidNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(PidNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.time_step = 1.

        self.P = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

        self.I = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

        self.D = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

        self.sum = nn.Sequential(
            nn.Linear(action_dim*3, 64),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

        self.apply(weight_init)

        self.prev_error_itgr = None
        self.prev_error = None

    def _format(self, error):
        if not isinstance(error, torch.Tensor):
            error = torch.tensor(error, device=self.device, dtype=torch.float32)
            error = error.unsqueeze(0)

        if self.prev_error_itgr is None:
            error_dot = error / self.time_step
            error_itgr = self.time_step*error
        else:
            error_dot = (error - self.prev_error) / self.time_step
            error_itgr = self.prev_error_itgr + self.time_step*error

        self.prev_error = error
        self.prev_error_itgr = error_itgr

        return error, error_itgr, error_dot

    def forward(self, error):
        error, error_itgr, error_dot = self._format(error)

        p_output = self.P(error)
        i_output = self.I(error_itgr)
        d_output = self.D(error_dot)

        pid_input = torch.cat([p_output, i_output, d_output], dim=-1)
        pid_output = self.sum(pid_input)
        pid_output = torch.tanh(pid_output)

        pid_rate = 1/(1 + torch.exp(-5*(torch.sqrt(torch.mean(error**2, dim=1, keepdim=True)) - 1)))
        pid_output = pid_rate*pid_output

        return pid_output

    def trains(self):
        self.P.train()
        self.I.train()
        self.D.train()
        self.sum.train()

    def evals(self):
        self.P.eval()
        self.I.eval()
        self.D.eval()
        self.sum.eval()


class InverseDynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, algorithm,
                 args, buffer=None, net_type=None, hidden_dim=256,
                 max_sigma=1e-2, min_sigma=1e-6, announce=True):
        super(InverseDynamicsNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.n_history = args.n_history

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.buffer4policy = algorithm.buffer
        self.buffer4model = buffer
        self.batch_size = 64

        self.train_cnt = 0

        if self.net_type == "DNN":
            # self.inv_dnmsNN_state_d = nn.ModuleList([nn.Linear(self.state_dim, int(hidden_dim/2)), nn.ReLU()]).cuda()
            # self.inv_dnmsNN_state = nn.ModuleList([nn.Linear(self.state_dim, int(hidden_dim/2)), nn.ReLU()]).cuda()
            # self.inv_dnmsNN = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            # self.inv_dnmsNN = self.inv_dnmsNN.append(nn.Linear(hidden_dim, action_dim)).cuda()

            self.inv_dnmsNN = nn.ModuleList(
                [nn.Linear((self.state_dim * 2 + self.action_dim) * self.n_history, hidden_dim), nn.Dropout(0.15),
                 nn.ReLU()])
            self.inv_dnmsNN = self.inv_dnmsNN.append(nn.Linear(hidden_dim, hidden_dim))
            self.inv_dnmsNN = self.inv_dnmsNN.append(nn.Dropout(0.15))
            self.inv_dnmsNN = self.inv_dnmsNN.append(nn.ReLU())
            self.inv_dnmsNN = self.inv_dnmsNN.append(nn.Linear(hidden_dim, action_dim)).cuda()

            self.inv_model_lr = args.inv_model_lr_dnn

        if self.net_type == "BNN":
            self.is_freeze = False

            self.inv_dnmsNN = nn.ModuleList([
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                in_features=self.state_dim * 2, out_features=hidden_dim), nn.ReLU()])
            self.inv_dnmsNN = self.inv_dnmsNN.append(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim))
            self.inv_dnmsNN = self.inv_dnmsNN.append(nn.ReLU())
            self.inv_dnmsNN = self.inv_dnmsNN.append(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=self.action_dim)
            ).cuda()

            self.inv_model_lr = args.inv_model_lr_bnn

        if self.net_type == "prob":

            self.mu = torch.zeros(action_dim)
            self.sigma = torch.zeros(action_dim)

            self.fc = nn.Linear(self.state_dim * 2, hidden_dim).cuda()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim).cuda()
            self.ln = nn.LayerNorm(hidden_dim).cuda()
            self.do1 = nn.Dropout(0.25)
            self.do2 = nn.Dropout(0.5)
            self.fc_mu = nn.Linear(hidden_dim, action_dim).cuda()
            self.fc_sigma = nn.Linear(hidden_dim, action_dim).cuda()

            self.max_sigma = max_sigma
            self.min_sigma = min_sigma
            assert (self.max_sigma >= self.min_sigma)
            if announce:
                print("Probabilistic transition model chosen.")

            self.inv_model_lr = args.inv_model_lr_dnn

        self.inv_dnms_optimizer = optim.AdamW(self.parameters(), lr=self.inv_model_lr)
        # self.scheduler = lr_scheduler.ExponentialLR(self.inv_dnms_optimizer, gamma=0.99)

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = args.inv_model_kl_weight

        self.apply(weight_init)

    def forward(self, state, next_state, prev_action, train=False):
        if train is False:
            self.inv_dnmsNN.eval()

        # BNN freezing part
        if self.net_type == 'BNN':
            if train is False and self.is_freeze is False:
                bnn.utils.eps_zero(self.inv_dnmsNN)
                self.is_freeze = True

            if train is True and self.is_freeze is True:
                bnn.utils.unfreeze(self.inv_dnmsNN)
                self.is_freeze = False

        # Tensorlizing
        if type(state) is not torch.Tensor:
            state = torch.tensor(state, dtype=torch.float).cuda()
        if type(next_state) is not torch.Tensor:
            next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        if type(prev_action) is not torch.Tensor:
            prev_action = torch.tensor(prev_action, dtype=torch.float).cuda()

        state = F.softsign(state)
        next_state = F.softsign(next_state)

        # state_d = (next_state - state)/(self.frameskip)

        # for i in range(len(self.inv_dnmsNN_state)):
        #     state_d = self.inv_dnmsNN_state_d[i](state_d)
        #     next_state = self.inv_dnmsNN_state[i](next_state)

        if train is True:
            z = torch.cat([state, next_state, prev_action], dim=1)
        else:
            z = torch.cat([state, next_state, prev_action])

        if self.net_type == "prob":
            z = self.fc(z)
            z = self.fc2(z)
            z = torch.relu(z)
            self.mu = self.fc_mu(z)
            if train is True:  # In training, sample output at normal distribution.
                # normalize fc_sigma output to 0 ~ 1 with sigmoid fcn
                sigma = torch.sigmoid(self.fc_sigma(z))
                # denormalize the sigma with min_sigma and max_sigma
                self.sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
                z = self.sample_prediction(self.mu, self.sigma)
            else:  # In testing, we use only mean value to inference a current action.
                z = self.mu
        else:  # Cases of DNN, BNN
            for i in range(len(self.inv_dnmsNN)):
                z = self.inv_dnmsNN[i](z)

        z = torch.tanh(z)
        return z

    def sample_prediction(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def train_all(self, training_num):
        self.inv_dnmsNN.train()

        self.train_cnt += 1

        cost = 0.0
        mse = 0.0
        kl = 0.0

        for i in range(training_num):

            if self.buffer4model is not None:
                # s_pn, a_pn, _, ns_pn, _ = self.buffer4policy.sample(int(self.batch_size/2))
                # s_mn, a_mn, _, ns_mn, _ = self.buffer4model.sample(int(self.batch_size/2))
                #
                # s = torch.cat([s_pn, s_mn])
                # a = torch.cat([a_pn, a_mn])
                # ns = torch.cat([ns_pn, ns_mn])
                s, a, _, ns, _ = self.buffer4model.sample(self.batch_size)
            else:
                s, a, _, ns, _ = self.buffer4model.sample(self.batch_size)

            z = self.forward(s, ns, a[:, self.action_dim:], train=True)

            if self.net_type == "prob":

                mse = self.mse_loss(z, a)

                error_mean = torch.reshape(torch.tanh(self.mu) - a, (-1, 1, self.action_dim))
                error_mean_trans = torch.transpose(error_mean, 1, 2)

                cost = torch.mean(
                    torch.reshape(
                        torch.matmul(torch.matmul(error_mean, torch.diag_embed(1 / self.sigma)), error_mean_trans),
                        (-1,)) \
                    + torch.reshape(torch.log(torch.prod(self.sigma, dim=1) + 1), (-1,))
                ) + self.l1_loss(z, a)

                kl = torch.Tensor(0)

            else:
                a_now = a[:, :self.action_dim]
                mse = self.mse_loss(z, a_now)
                kl = self.kl_loss(self.inv_dnmsNN)
                if self.train_cnt < 10000:
                    cost = mse + self.l1_loss(z, a_now)
                else:
                    cost = mse + self.kl_weight * kl + self.l1_loss(z, a_now)

            self.inv_dnms_optimizer.zero_grad()
            cost.backward()
            self.inv_dnms_optimizer.step()
        # self.scheduler.step()

        cost = cost.cpu().detach().numpy()
        mse = mse.cpu().detach().numpy()
        kl = kl.cpu().detach().numpy()
        return cost, mse, kl

    def eval_model(self, state, action, next_state):
        self.inv_dnmsNN.eval()

        state = torch.tensor(state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        # state_d = (next_state - state)/self.frameskip

        z = self.forward(state, next_state, action[self.action_dim:])
        error = torch.max(torch.abs(z - action[:self.action_dim]))
        error = error.cpu().detach().numpy()

        return error


if __name__ == '__main__':
    pass
