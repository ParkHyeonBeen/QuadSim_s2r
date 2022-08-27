import argparse
import numpy as np
import torch


parser = argparse.ArgumentParser(description="QuadSim_save_param")
parser.add_argument("--base-path", default="/home/phb/ETRI/QuadSim_s2r/results/", type=str, help="base path of the current project")
parser.add_argument("--result-name", default="0825-0423QuadRotor-v0", type=str, help="Checkpoint path to a pre-trained model.")
args = parser.parse_args()

if __name__ == '__main__':
    model_path = args.base_path+args.result_name+'/network/policy/'
    model = torch.load(model_path + "policy_best")
    w1 = model['linear1.weight'].cpu().numpy()
    b1 = model['linear1.bias'].cpu().numpy()
    w2 = model['linear2.weight'].cpu().numpy()
    b2 = model['linear2.bias'].cpu().numpy()
    mean_w = model['mean_linear.weight'].cpu().numpy()
    mean_b = model['mean_linear.bias'].cpu().numpy()
    # std_w = model['log_std_linear.weight'].cpu().numpy()
    # std_b =model['log_std_linear.bias'].cpu().numpy()

    np.save(model_path + 'w1', w1)
    np.save(model_path + 'b1', b1)
    np.save(model_path + 'w2', w2)
    np.save(model_path + 'b2', b2)
    np.save(model_path + 'w3', mean_w)
    np.save(model_path + 'b3', mean_b)
    # np.save(args.model_path + 'log_std_w', std_w)
    # np.save(args.model_path + 'log_std_b', std_b)
