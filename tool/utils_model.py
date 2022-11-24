from tool.utils import *
import torch_ard as nn_ard

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
            network_state = np.concatenate([state["position_error_obs"],
                                            state["velocity_error_obs"],
                                            state["rotation_obs"],
                                            state["angular_velocity_error_obs"]])

            next_network_state = np.concatenate([next_state["position_error_obs"][:3],
                                                 next_state["velocity_error_obs"][:3],
                                                 next_state["rotation_obs"][:6],
                                                 next_state["angular_velocity_error_obs"][:3]])
            prev_network_action = state["action_obs"][env.action_dim:]

    return network_state, prev_network_action, next_network_state

def compressing(network):
    print("--------compressing---------------")

    compressed_model = {}

    network.state_net[0].save_net()
    network.prev_action_net[0].save_net()
    network.middle_net[0].save_net()
    network.next_state_net[0].save_net()
    network.action_net[0].save_net()
    network.action_net[2].save_net()

    compressed_model["state_net.0.weight"] = network.state_net[0].W.detach()
    compressed_model["prev_action_net.0.weight"] = network.prev_action_net[0].W.detach()
    compressed_model["middle_net.0.weight"] = network.middle_net[0].W.detach()
    compressed_model["next_state_net.0.weight"] = network.next_state_net[0].W.detach()
    compressed_model["action_net.0.weight"] = network.action_net[0].W.detach()
    compressed_model["action_net.2.weight"] = network.action_net[2].W.detach()

    compressed_model["state_net.0.bias"] = network.state_net[0].bias.detach()
    compressed_model["prev_action_net.0.bias"] = network.prev_action_net[0].bias.detach()
    compressed_model["middle_net.0.bias"] = network.middle_net[0].bias.detach()
    compressed_model["next_state_net.0.bias"] = network.next_state_net[0].bias.detach()
    compressed_model["action_net.0.bias"] = network.action_net[0].bias.detach()
    compressed_model["action_net.2.bias"] = network.action_net[2].bias.detach()

    return compressed_model

def save_model(network, loss_best, loss_now, path, ard=False):

    print("--------save model ---------------")
    save_state = {}

    if network.net_type == "bnn":
        save_state["network"] = compressing(network)
        sparsity_ratio = round(100. * nn_ard.get_dropped_params_ratio(network), 3)
        save_state["sparsity_ratio"] = sparsity_ratio
        print('Sparsification ratio: %.3f%%' % sparsity_ratio)
    else:
        save_state["network"] = network.state_dict()
        # save_state = {"state_net.0.weight": network.state_dict()["state_net.0.weight"],
        #               "state_net.0.bias": network.state_dict()["state_net.0.bias"],
        #               "prev_action_net.0.weight": network.state_dict()["prev_action_net.0.weight"],
        #               "prev_action_net.0.bias": network.state_dict()["prev_action_net.0.bias"],
        #               "middle_net.0.weight": network.state_dict()["middle_net.0.weight"],
        #               "middle_net.0.bias": network.state_dict()["middle_net.0.bias"],
        #               "next_state_net.0.weight": network.state_dict()["next_state_net.0.weight"],
        #               "next_state_net.0.bias": network.state_dict()["next_state_net.0.bias"],
        #               "action_net.0.weight": network.state_dict()["action_net.0.weight"],
        #               "action_net.0.bias": network.state_dict()["action_net.0.bias"],
        #               "action_net.2.weight": network.state_dict()["action_net.2.weight"],
        #               "action_net.2.bias": network.state_dict()["action_net.2.bias"]
        #               }

    if loss_best > loss_now:
        if ard:
            torch.save(save_state, path + "/best_" + path[-3:])
        else:
            torch.save(save_state, path + "/better_" + path[-3:])
        return loss_now
    else:
        if not ard:
            torch.save(save_state, path + "/current_" + path[-3:])

def get_action(policy_net, state):
    network_state = np.concatenate([state["position_error_obs"],
                                    state["velocity_error_obs"],
                                    state["rotation_obs"],
                                    state["angular_velocity_error_obs"]])

    return policy_net.get_action(network_state, deterministic=True)

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