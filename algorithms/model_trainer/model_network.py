import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbnn as bnn
from tool.utils import *

def _format(device, *inp):
    output = []
    for d in inp:
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, device=device, dtype=torch.float32)
            d = d.unsqueeze(0)
        output.append(d)
    return output

class ModelNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, args, net_type=None):
        super(ModelNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.args = args

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_history = self.args.n_history

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args.batch_size

        if self.net_type == "dnn":
            self.state_net = nn.Sequential(
                nn.Linear(self.state_dim*self.n_history, int(self.hidden_dim/2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.action_net = nn.Sequential(
                nn.Linear(self.action_dim * 2, int(self.hidden_dim/2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.next_state_net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.Dropout(0.15),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.state_dim)
            )

        if self.net_type == "bnn":
            self.state_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                in_features=self.state_dim * self.n_history, out_features=int(self.hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.action_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                in_features=self.action_dim * 2, out_features=int(self.hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.next_state_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=int(self.hidden_dim / 2), out_features=int(self.hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU(),
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.hidden_dim, out_features=self.state_dim)
            )

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = args.model_kl_weight

        self.apply(weight_init)

    def forward(self, state, action):

        out = _format(self.device, state, action)

        state = self.state_net(out[0])
        action = self.action_net(out[1])

        next_state = self.next_state_net(torch.cat([state, action], dim=-1))

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
    def __init__(self, state_dim, action_dim, hidden_dim, args):
        super(PidNetwork, self).__init__()

        self.args = args

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_history = self.args.n_history
        if args.gpu:
            self.device = torch.device("cuda:" + str(args.device_idx) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.time_step = 1.


        self.P = nn.Sequential(
            nn.Linear(self.state_dim, 64),
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

class InverseModelNetwork(nn.Module):
    def __init__(self, env, hidden_dim, args, net_type=None):
        super(InverseModelNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.args = args

        self.action_dim = env.action_dim
        self.position_dim = env.position_dim
        self.velocity_dim = env.velocity_dim
        self.rotation_dim = env.rotation_dim
        self.hidden_dim = hidden_dim

        self.state_net_input = (self.position_dim + self.rotation_dim)*2*self.args.n_history
        self.next_state_net_input = self.position_dim + self.rotation_dim
        self.prev_action_net_input = self.action_dim * (self.args.n_history-1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch_size = args.batch_size

        self.train_cnt = 0
        self.KL_train_start = 10000

        # Regularization tech
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.do1 = nn.Dropout(0.15)

        # construct the structure of model network
        if self.net_type == "dnn":
            self.state_net = nn.Sequential(
                nn.Linear(self.state_net_input, int(self.hidden_dim/2)),
                nn.Dropout(0.05),
                nn.ReLU(),
                nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2))

            )
            self.prev_action_net = nn.Sequential(
                nn.Linear(self.prev_action_net_input, int(self.hidden_dim / 2)),
                # nn.Dropout(0.05),
                nn.ReLU(),
                # nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2))
            )
            self.middle_net = nn.Sequential(
                nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)),
                # nn.Dropout(0.05),
                nn.ReLU(),
                # nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2))
            )
            self.next_state_net = nn.Sequential(
                nn.Linear(self.next_state_net_input, int(self.hidden_dim/2)),
                # nn.Dropout(0.05),
                nn.ReLU(),
                # nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2))
            )
            self.action_net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(0.05),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim)
            )

        if self.net_type == "bnn":
            self.is_freeze = False

            self.state_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.05,
                                in_features=self.state_net_input, out_features=int(self.hidden_dim/2)),
                # nn.Dropout(0.15),
                nn.ReLU()
            )

            self.prev_action_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.05,
                                in_features=self.prev_action_net_input, out_features=int(self.hidden_dim / 2)),
                # nn.Dropout(0.15),
                nn.ReLU()
            )

            self.middle_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.05,
                                in_features=self.hidden_dim, out_features=int(self.hidden_dim / 2)),
                # nn.Dropout(0.15),
                nn.ReLU()
            )

            self.next_state_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.05,
                                in_features=self.next_state_net_input, out_features=int(self.hidden_dim/2)),
                # nn.Dropout(0.15),
                nn.ReLU()
            )

            self.action_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.05,
                                in_features=self.hidden_dim, out_features=self.hidden_dim),
                # nn.Dropout(0.15),
                nn.ReLU(),
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.05,
                                in_features=self.hidden_dim, out_features=self.action_dim)
            )

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

        self.kl_weight = args.inv_model_kl_weight

        self.apply(weight_init)

    def forward(self, state, prev_action, next_state, train=False):

        # bnn freezing part to evaluate the model with mean value
        if self.net_type == 'bnn':
            if train is False and self.is_freeze is False:
                # eps_zero fcn is a customized fcn to make eps weight and bias be zero.
                # so, you should customize utils.__init__.py , freeze_model.py, and linear.py
                bnn.utils.eps_zero(self.state_net)
                bnn.utils.eps_zero(self.prev_action_net)
                bnn.utils.eps_zero(self.middle_net)
                bnn.utils.eps_zero(self.next_state_net)
                bnn.utils.eps_zero(self.action_net)
                self.is_freeze = True

            if train is True and self.is_freeze is True:
                bnn.utils.unfreeze(self.state_net)
                bnn.utils.unfreeze(self.prev_action_net)
                bnn.utils.unfreeze(self.middle_net)
                bnn.utils.unfreeze(self.next_state_net)
                bnn.utils.unfreeze(self.action_net)
                self.is_freeze = False

        # Tensorlizing
        out = _format(self.device, state, prev_action, next_state)

        # To normalize(-1~1) input data by using the softsign fcn
        state = self.state_net(out[0])
        prev_action = self.prev_action_net(out[1])

        middle = self.middle_net(torch.cat([state, prev_action], dim=-1))
        next_state = self.next_state_net(out[2])

        # Testing data has 1 dim date(vector), so we should remove dim=1
        action = torch.tanh(self.action_net(torch.cat([middle, next_state], dim=-1)))
        return action

    def trains(self):
        self.state_net.train()
        self.action_net.train()
        self.prev_action_net.train()
        self.middle_net.train()
        self.next_state_net.train()

    def evals(self):
        self.state_net.eval()
        self.action_net.eval()
        self.prev_action_net.eval()
        self.middle_net.eval()
        self.next_state_net.eval()

    # def update(self):
    #     self.imn.train()
    #     self.train_cnt += 1
    #
    #     s, a, _, ns, _ = self.buffer4policy.sample(self.batch_size)
    #     a = a[:, :self.action_dim]
    #
    #     z = self.forward(s, ns, train=True)
    #
    #     mse = self.mse_loss(z, a)
    #     kl = self.kl_loss(self.imn)
    #
    #     # After a KL-loss start step, training related to KL loss is started
    #     if self.net_type == 'bnn':
    #         if self.train_cnt < self.KL_train_start:
    #             cost = mse + self.l1_loss(z, a)
    #         else:
    #             cost = mse + self.kl_weight * kl + self.l1_loss(z, a)
    #     else:
    #         cost = mse + self.l1_loss(z, a)
    #
    #     self.imn_optimizer.zero_grad()
    #     cost.backward()
    #     self.imn_optimizer.step()
    #
    #     cost = cost.cpu().detach().numpy()
    #     mse = mse.cpu().detach().numpy()
    #     kl = kl.cpu().detach().numpy()
    #     return cost, mse, kl
    #
    # def evaluate(self, state, action, next_state):
    #     self.imn.eval()
    #     with torch.no_grad():
    #         state = torch.tensor(state, dtype=torch.float).cuda()
    #         action = torch.tensor(action, dtype=torch.float).cuda()
    #         next_state = torch.tensor(next_state, dtype=torch.float).cuda()
    #
    #         z = self.forward(state, next_state)
    #
    #     error = torch.max(torch.abs(z - action)).cpu().numpy()
    #
    #     return error


