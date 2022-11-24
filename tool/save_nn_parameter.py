import argparse
import numpy as np
import torch


parser = argparse.ArgumentParser(description="QuadSim_save_param")
parser.add_argument("--base-path", default="/home/phb/ETRI/QuadSim_s2r/results/", type=str, help="base path of the current project")
parser.add_argument("--result-name", default="1108-1627QuadRotor-v0", type=str, help="Checkpoint path to a pre-trained model.")
parser.add_argument("--which_model", default="both", type=str, help="both: policy and model"
                                                                    "policy: only policy"
                                                                    "policy: only model")
parser.add_argument("--net_type", default="dnn", type=str, help="dnn, bnn")


args = parser.parse_args()

def save_policy_to_npy(policy, path):
    w1 = policy['linear1.weight'].cpu().numpy()
    b1 = policy['linear1.bias'].cpu().numpy()
    w2 = policy['linear2.weight'].cpu().numpy()
    b2 = policy['linear2.bias'].cpu().numpy()
    mean_w = policy['mean_linear.weight'].cpu().numpy()
    mean_b = policy['mean_linear.bias'].cpu().numpy()
    # std_w = model['log_std_linear.weight'].cpu().numpy()
    # std_b =model['log_std_linear.bias'].cpu().numpy()

    np.save(path + 'w1', w1)
    np.save(path + 'b1', b1)
    np.save(path + 'w2', w2)
    np.save(path + 'b2', b2)
    np.save(path + 'w3', mean_w)
    np.save(path + 'b3', mean_b)
    # np.save(args.policy_path + 'log_std_w', std_w)3ㅈㅋ
    # np.save(args.policy_path + 'log_std_b', std_b)

def _get_param_dnn(model, network_name, num_layer, path):
    for i in range(num_layer):
        network_w_mu = model[network_name + '_net.' + str(2*i) + '.weight'].cpu().numpy()
        network_b_mu = model[network_name + '_net.' + str(2*i) + '.bias'].cpu().numpy()

        np.save(path + 'w_mu_' + network_name, network_w_mu)
        np.save(path + 'b_mu_' + network_name, network_b_mu)

def _get_param_bnn(model, network_name, num_layer, path):
    for i in range(num_layer):
        network_w_mu = model[network_name + '_net.' + str(2*i) + '.weight'].cpu().numpy()
        network_b_mu = model[network_name + '_net.' + str(2*i) + '.bias'].cpu().numpy()

        np.save(path + 'w_mu_' + network_name, network_w_mu)
        np.save(path + 'b_mu_' + network_name, network_b_mu)

    # for i in range(num_layer):
    #     print(model.keys())
    #     network_w_mu = model[network_name + '_net.' + str(2*i) + '.weight_mu'].cpu().numpy()
    #     network_b_mu = model[network_name + '_net.' + str(2*i) + '.bias_mu'].cpu().numpy()
    #     network_w_std = np.exp(model[network_name + '_net.' + str(2*i) + '.weight_log_sigma'].cpu().numpy())
    #     network_b_std = np.exp(model[network_name + '_net.' + str(2*i) + '.bias_log_sigma'].cpu().numpy())
    #
    #     if num_layer != 1:
    #         np.save(path + 'w_mu_' + str(i) + "_" + network_name, network_w_mu)
    #         np.save(path + 'b_mu_' + str(i) + "_" + network_name, network_b_mu)
    #         np.save(path + 'w_std_' + str(i) + "_" + network_name, network_w_std)
    #         np.save(path + 'b_std_' + str(i) + "_" + network_name, network_b_std)
    #     else:
    #         np.save(path + 'w_mu_' + network_name, network_w_mu)
    #         np.save(path + 'b_mu_' + network_name, network_b_mu)
    #         np.save(path + 'w_std_' + network_name, network_w_std)
    #         np.save(path + 'b_std_' + network_name, network_b_std)

def save_model_to_npy(model, net_list, path):
    if "dnn" in path:
        for net in net_list:
            _get_param_dnn(model, net[0], net[1], path)

    if "bnn" in path:
        for net in net_list:
            _get_param_bnn(model, net[0], net[1], path)


if __name__ == '__main__':
    network_path = args.base_path+args.result_name+'/network/'
    policy_path = network_path + "policy/"
    model_path = network_path + "model/" + args.net_type + "/"
    policy = torch.load(policy_path + "policy_better")
    model = torch.load(model_path + "better_" + args.net_type)
    print(model.keys())
    print(model)

    net_list = [("state", 1),
                ("prev_action", 1),
                ("middle", 1),
                ("next_state", 1),
                ("action", 2)]

    if args.which_model == "both":
        save_policy_to_npy(policy, policy_path)
        save_model_to_npy(model, net_list, model_path)

    if args.which_model == "policy":
        save_policy_to_npy(policy, policy_path)

    if args.which_model == "model":
        save_model_to_npy(model, net_list, model_path)