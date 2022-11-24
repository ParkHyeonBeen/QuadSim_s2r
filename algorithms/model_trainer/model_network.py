import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbnn as bnn
from tool.utils import *
import torch_ard as nn_ard
import torch_rbf as nn_rbf

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

        if args.develop_version == 1:
            self.state_net_input = (self.position_dim + self.rotation_dim) * 2 * self.args.n_history
            self.next_state_net_input = self.position_dim + self.rotation_dim
        else:
            self.state_net_input = env.state_dim*self.args.n_history
            self.next_state_net_input = env.state_dim

        self.prev_action_net_input = self.action_dim * (self.args.n_history-1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # # Regularization tech
        # self.ln = nn.LayerNorm(self.hidden_dim)
        # self.bn = nn.BatchNorm1d(self.hidden_dim)
        # self.do = nn.Dropout(0.05)

        # construct the structure of model network
        if self.net_type == "dnn":
            self.state_net = nn.Sequential(
                nn.Linear(self.state_net_input, int(self.hidden_dim/2)),
                nn.ReLU(),
            )
            self.prev_action_net = nn.Sequential(
                nn.Linear(self.prev_action_net_input, int(self.hidden_dim / 2)),
                nn.ReLU(),
            )
            self.middle_net = nn.Sequential(
                nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)),
                nn.ReLU(),
            )
            self.next_state_net = nn.Sequential(
                nn.Linear(self.next_state_net_input, int(self.hidden_dim/2)),
                nn.ReLU(),
            )
            self.action_net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim)
            )

        if self.net_type == "bnn":

            self.state_net = nn.Sequential(
                nn_ard.LinearARD(in_features=self.state_net_input, out_features=int(self.hidden_dim/2)),
                nn.ReLU()
            )

            self.prev_action_net = nn.Sequential(
                nn_ard.LinearARD(in_features=self.prev_action_net_input, out_features=int(self.hidden_dim / 2)),
                nn.ReLU()
            )

            self.middle_net = nn.Sequential(
                nn_ard.LinearARD(in_features=self.hidden_dim, out_features=int(self.hidden_dim / 2)),
                nn.ReLU()
            )

            self.next_state_net = nn.Sequential(
                nn_ard.LinearARD(in_features=self.next_state_net_input, out_features=int(self.hidden_dim/2)),
                nn.ReLU()
            )

            self.action_net = nn.Sequential(
                nn_ard.LinearARD(in_features=self.hidden_dim, out_features=self.hidden_dim),
                nn.ReLU(),
                nn_ard.LinearARD(in_features=self.hidden_dim, out_features=self.action_dim)
            )

        if self.net_type == "rbf":

            network_input = self.state_net_input + self.prev_action_net_input + self.next_state_net_input

            self.action_net = nn.Sequential(
                nn_rbf.RBF(in_features=network_input, out_features=self.hidden_dim, basis_func=nn_rbf.gaussian),
                nn.Linear(in_features=self.hidden_dim, out_features=self.action_dim),
            )

            # self.state_net = nn.Sequential(
            #     nn_rbf.RBF(in_features=self.state_net_input, out_features=int(self.hidden_dim/2), basis_func=nn_rbf.gaussian),
            #     nn.Linear(in_features=int(self.hidden_dim/2), out_features=int(self.hidden_dim/2)),
            # )
            #
            # self.prev_action_net = nn.Sequential(
            #     nn_rbf.RBF(in_features=self.prev_action_net_input, out_features=int(self.hidden_dim/2), basis_func=nn_rbf.gaussian),
            #     nn.Linear(in_features=int(self.hidden_dim/2), out_features=int(self.hidden_dim/2)),
            # )
            #
            # self.middle_net = nn.Sequential(
            #     nn_rbf.RBF(in_features=self.hidden_dim, out_features=int(self.hidden_dim/2), basis_func=nn_rbf.gaussian),
            #     nn.Linear(in_features=int(self.hidden_dim/2), out_features=int(self.hidden_dim/2)),
            # )
            #
            # self.next_state_net = nn.Sequential(
            #     nn_rbf.RBF(in_features=self.next_state_net_input, out_features=int(self.hidden_dim/2), basis_func=nn_rbf.gaussian),
            #     nn.Linear(in_features=int(self.hidden_dim/2), out_features=int(self.hidden_dim/2)),
            # )
            #
            # self.action_net = nn.Sequential(
            #     nn_rbf.RBF(in_features=self.hidden_dim, out_features=self.hidden_dim, basis_func=nn_rbf.gaussian),
            #     nn.Linear(in_features=self.hidden_dim, out_features=self.action_dim),
            # )

        self.apply(weight_init)

    def forward(self, state, prev_action, next_state):

        # Tensorlizing
        out = _format(self.device, state, prev_action, next_state)

        if self.net_type == 'rbf':
            action = self.action_net(torch.cat(out, dim=-1))
        else:
            state = self.state_net(out[0])
            prev_action = self.prev_action_net(out[1])

            middle = self.middle_net(torch.cat([state, prev_action], dim=-1))
            next_state = self.next_state_net(out[2])

            action = torch.tanh(self.action_net(torch.cat([middle, next_state], dim=-1)))
        return action

    def trains(self):
        self.action_net.train()

        if self.net_type != 'rbf':
            self.state_net.train()
            self.prev_action_net.train()
            self.middle_net.train()
            self.next_state_net.train()

    def evaluates(self):
        self.action_net.eval()

        if self.net_type != 'rbf':
            self.state_net.eval()
            self.prev_action_net.eval()
            self.middle_net.eval()
            self.next_state_net.eval()

class CompressedInverseModelNetwork(nn.Module):
    def __init__(self, env, hidden_dim, args, net_type=None):
        super(CompressedInverseModelNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.state_net = nn.Sequential(
            nn.Linear(self.state_net_input, int(self.hidden_dim / 2)),
            nn.ReLU(),
        )
        self.prev_action_net = nn.Sequential(
            nn.Linear(self.prev_action_net_input, int(self.hidden_dim / 2)),
            nn.ReLU(),
        )
        self.middle_net = nn.Sequential(
            nn.Linear(self.hidden_dim, int(self.hidden_dim / 2)),
            nn.ReLU(),
        )
        self.next_state_net = nn.Sequential(
            nn.Linear(self.next_state_net_input, int(self.hidden_dim / 2)),
            nn.ReLU(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

    def forward(self, state, prev_action, next_state):

        # Tensorlizing
        out = _format(self.device, state, prev_action, next_state)

        if self.net_type == 'rbf':
            action = self.action_net(torch.cat(out, dim=-1))
        else:
            state = self.state_net(out[0])
            prev_action = self.prev_action_net(out[1])

            middle = self.middle_net(torch.cat([state, prev_action], dim=-1))
            next_state = self.next_state_net(out[2])

            action = torch.tanh(self.action_net(torch.cat([middle, next_state], dim=-1)))
        return action

# class CompressedInverseModelNetwork(nn.Module):
#     def __init__(self, saved_network):
#         super(CompressedInverseModelNetwork, self).__init__()
#
#         self.state_net_weight = saved_network["state_net.0.weight"]
#         self.state_net_bias = saved_network["state_net.0.bias"]
#         self.prev_action_net_weight = saved_network["prev_action_net.0.weight"]
#         self.prev_action_net_bias = saved_network["prev_action_net.0.bias"]
#         self.middle_net_weight = saved_network["middle_net.0.weight"]
#         self.middle_net_bias = saved_network["middle_net.0.bias"]
#         self.next_state_net_weight = saved_network["next_state_net.0.weight"]
#         self.next_state_net_bias = saved_network["next_state_net.0.bias"]
#         self.action_net_weight = saved_network["action_net.0.weight"]
#         self.action_net_bias = saved_network["action_net.0.bias"]
#         self.action_net_weight2 = saved_network["action_net.2.weight"]
#         self.action_net_bias2 = saved_network["action_net.2.bias"]
#
#     def forward(self, state, prev_action, next_state):
#
#         # Tensorlizing
#         out = _format(self.device, state, prev_action, next_state)
#
#         if self.net_type == 'rbf':
#             action = F.relu(F.linear(torch.cat(out, dim=-1), self.action_net_weight) + self.action_net_bias)
#             action = torch.tanh(F.linear(action, self.action_net_weight2) + self.action_net_bias2)
#         else:
#             state = F.relu(F.linear(out[0], self.state_net_weight) + self.state_net_bias)
#             prev_action = F.relu(F.linear(out[1], self.prev_action_net_weight) + self.prev_action_net_bias)
#
#             middle = F.relu(F.linear(torch.cat([state, prev_action], dim=-1), self.middle_net_weight) + self.middle_net_bias)
#             next_state = F.relu(F.linear(out[3], self.next_state_net_weight) + self.next_state_net_bias)
#
#             action = F.relu(F.linear(torch.cat([middle, next_state], dim=-1), self.action_net_weight) + self.action_net_bias)
#             action = torch.tanh(F.linear(action, self.action_net_weight2) + self.action_net_bias2)
#
#         return action