from tool.utils import *
import torch_ard as nn_ard

# def save_model(network, fname : str, path : str):
#     if "dnn" in fname:
#         torch.save(network.state_dict(), path + fname)
#     elif "bnn" in fname:
#         torch.save(network.state_dict(), path + fname)
#     else:
#         torch.save(network.state_dict(), path + fname)

def get_model_net_input(env, state, next_state=None, ver=0):
    # "next_state is None" means Train mode

    if ver == 1:
        if next_state is None:
            network_state = np.concatenate([state["position_obs"],
                                            (state["position_next_obs"] - state["position_obs"])/env.sample_time,
                                            state["rotation_obs"],
                                            (state["rotation_next_obs"] - state["rotation_obs"])/env.sample_time], axis=1)
            next_network_state = np.concatenate([state["position_next_obs"][:, :3],
                                                 state["rotation_next_obs"][:, :6]], axis=1)
            prev_network_action = state["action_obs"][:, env.action_dim:]
        else:
            network_state = np.concatenate([state["position_obs"],
                                            (next_state["position_obs"] - state["position_obs"]) / env.sample_time,
                                            state["rotation_obs"],
                                            (next_state["rotation_obs"] - state["rotation_obs"]) / env.sample_time], axis=-1)
            next_network_state = np.concatenate([next_state["position_obs"][:3],
                                                 next_state["rotation_obs"][:6]], axis=-1)
            prev_network_action = state["action_obs"][env.action_dim:]

    else:
        if next_state is None:
            network_state = np.concatenate([state["position_error_obs"],
                                            state["velocity_error_obs"],
                                            state["rotation_obs"],
                                            state["angular_velocity_error_obs"]], axis=1)
            next_network_state = np.concatenate([state["position_error_next_obs"][:, :3],
                                                 state["velocity_error_next_obs"][:, :3],
                                                 state["rotation_next_obs"][:, :6],
                                                 state["angular_velocity_error_next_obs"][:, :3]], axis=1)
            prev_network_action = state["action_obs"][:, env.action_dim:]
        else:
            network_state = None
            next_network_state = np.concatenate([next_state["position_error_obs"][:3],
                                                 next_state["velocity_error_obs"][:3],
                                                next_state["rotation_obs"][:6],
                                                next_state["angular_velocity_error_obs"][:3]])
            prev_network_action = state["action_obs"][env.action_dim:]

    return network_state, prev_network_action, next_network_state

def create_models(state_dim, action_dim, algorithm, args, net_type="dnn,bnn"):

    models = {}

    dnn = True if "dnn" in net_type else False
    bnn = True if "bnn" in net_type else False

    if dnn and args.develop_mode != 'DeepDOB':
        models['ModelNetDnn'] = DynamicsNetwork(state_dim, action_dim, args, net_type="DNN")
    if dnn is True and args.develop_mode != 'MRAP':
        models['InvModelNetDnn'] = InverseDynamicsNetwork(state_dim, action_dim, algorithm, args, net_type="DNN")
    if bnn is True and args.develop_mode != 'DeepDOB':
        models['ModelNetBnn'] = DynamicsNetwork(state_dim, action_dim, args, net_type="BNN")
    if bnn is True and args.develop_mode != 'MRAP':
        models['InvModelNetBnn'] = InverseDynamicsNetwork(state_dim, action_dim, algorithm, args, net_type="BNN")

    return models

def eval_models(state, action, next_state, models):
    error_list = []
    for model in models.values:
        _error = model.eval_model(state, action, next_state)
        error_list.append(_error)
    errors = np.hstack(error_list)
    return errors

def train_alls(training_step, models):
    cost_list = []
    mse_list = []
    kl_list = []

    if len(models) == 0:
        raise Exception("your models is empty now")

    for model in models.values():
        _cost, _mse, _kl = model.train_all(training_step)
        cost_list.append(_cost)
        mse_list.append(_mse)
        kl_list.append(_kl)

    costs = np.hstack(cost_list)
    mses = np.hstack(mse_list)
    kls = np.hstack(kl_list)

    return costs, mses, kls

def save_model(network, loss_best, loss_now, path, ard=False):
    if loss_best > loss_now:
        if ard:
            torch.save(network.state_dict(), path + "/best_" + path[-3:])
        else:
            torch.save(network.state_dict(), path + "/better_" + path[-3:])
    else:
        if not ard:
            torch.save(network.state_dict(), path + "/current_" + path[-3:])

def load_models(args_tester, model):

    path = args_tester.path
    path_model = None
    path_invmodel = None

    if "DNN" in args_tester.modelnet_name:
        if args_tester.prev_result is True:
            path_model = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/DNN/" + args_tester.modelnet_name
            path_invmodel = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/DNN/inv" + args_tester.modelnet_name
        else:
            path_model = path + args_tester.result_fname + "saved_net/model/DNN/" + args_tester.modelnet_name
            path_invmodel = path + args_tester.result_fname + "saved_net/model/DNN/inv" + args_tester.modelnet_name

    if "BNN" in args_tester.modelnet_name:
        if args_tester.prev_result is True:
            path_model = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/BNN/" + args_tester.modelnet_name
            path_invmodel = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/BNN/inv" + args_tester.modelnet_name
        else:
            path_model = path + args_tester.result_fname + "saved_net/model/BNN/" + args_tester.modelnet_name
            path_invmodel = path + args_tester.result_fname + "saved_net/model/BNN/inv" + args_tester.modelnet_name

    if args_tester.develop_mode == "MRAP":
        model.load_state_dict(torch.load(path_model))
    if args_tester.develop_mode == "DeepDOB":
        if "bnn" in args_tester.result_fname:
            model_tmp = torch.load(path_invmodel)
            for key in model_tmp.copy().keys():
                if 'eps' in key:
                    del(model_tmp[key])
            model.load_state_dict(model_tmp)
        else:
            model.load_state_dict(torch.load(path_invmodel))

def validate_measure(error_list):
    error_max = np.max(error_list, axis=0)
    mean = np.mean(error_list, axis=0)
    std = np.std(error_list, axis=0)
    loss = np.sqrt(mean**2 + std**2)

    return [loss, mean, std, error_max]

def get_random_action_batch(observation, env_action, test_env, model_buffer, max_action, min_action):

    env_action_noise, _ = add_noise(env_action, scale=0.1)
    action_noise = normalize(env_action_noise, max_action, min_action)
    next_observation, reward, done, info = test_env.step(env_action_noise)
    model_buffer.add(observation, action_noise, reward, next_observation, float(done))

def set_sync_env(env, test_env):

    position = env.sim.data.qpos.flat.copy()
    velocity = env.sim.data.qvel.flat.copy()

    test_env.set_state(position, velocity)