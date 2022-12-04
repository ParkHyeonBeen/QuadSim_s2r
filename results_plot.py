import argparse, sys, os
from copy import deepcopy
from tool.utils import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Results integrated plot')

parser.add_argument('--base_path', default="/home/phb/ETRI/QuadSim_s2r/", help='base path of the current project')
parser.add_argument("--env_name", "-en", default="QuadRotor-v0", type=str, help="the name of environment to show")
parser.add_argument("--from_csv", "-fc", default="True", type=str2bool, help="If True, you will get the results from csv file")
parser.add_argument("--each_plot", "-ep", default="False", type=str2bool, help="If True, we can get each plots")
parser.add_argument('--max_disturb', '-xd', default=40, type=float, help='QuadRotor   : 20')
parser.add_argument('--min_disturb', '-nd', default=0.0, type=float, help='')
parser.add_argument('--max_uncertain', '-xu', default=50, type=float, help='quadrotor   : 80')

parser.add_argument('--max_noise', '-xn', default=0.1, type=float, help='max std of gaussian noise for state')
parser.add_argument('--min_noise', '-nn', default=0.0, type=float, help='min std of gaussian noise for state')

parser.add_argument('--label_size', '-bs', default=15., type=float, help='XY label size')
parser.add_argument('--legend_size', '-gs', default=10., type=float, help='Legend size')
parser.add_argument('--window_size', '-ws', default=(10, 8), help='min std of gaussian noise for state')

args = parser.parse_known_args()

if len(args) != 1:
    args = args[0]

legend_list = ["Only Policy network",
               "Policy network with DNN-based DOB",
               "Policy network with SBN-based DOB"]

xlabel_list = ["Percentage of disturbance magnitude over action range [%]",
               "Model uncertainty ratio [%]",
               "Standard deviation of Gaussian noise"]

ylabel_list = ["Average return over 100 times",
               "Success ratio over 100 times",
               "Mean square error of estimation error"]

def get_env_file(results_list):
    specific_file = []

    for result_name in results_list:
        if args.env_name in result_name:
            specific_file.append(result_name)

    return specific_file

def find_each_results(results_list):
    results_npy = OrderedDict()

    results_npy["reward"] = OrderedDict()
    results_npy["success_rate"] = OrderedDict()
    results_npy["model_error"] = OrderedDict()

    for results in results_list:
        result_path = "./results/" + results + "/log/test/"
        test_log_list = os.listdir(result_path)
        for test_log in test_log_list:
            if "csv" in test_log and args.from_csv is True:
                terms = test_log[:-4].split("_")

                indicator_name = terms[0] if "reward" in terms else "_".join(terms[:2])
                type_name = "none" if "none" in terms else terms[-2]
                case_name = terms[-1]

                results_npy[indicator_name][case_name] = results_npy[indicator_name].get(case_name, OrderedDict())
                results_npy[indicator_name][case_name][type_name] = np.loadtxt(result_path + test_log,
                                                                               skiprows=1,
                                                                               delimiter=',',
                                                                               dtype='float')[:, 1:]
            else:
                if "npy" in test_log:
                    terms = test_log[:-4].split("_")

                    indicator_name = terms[0] if "reward" in terms else "_".join(terms[:2])
                    type_name = "none" if "none" in terms else terms[-2]
                    case_name = terms[-1]

                    results_npy[indicator_name][case_name] = results_npy[indicator_name].get(case_name, OrderedDict())
                    results_npy[indicator_name][case_name][type_name] = np.load(result_path + test_log)

    for indicator_val in results_npy.values():
        for case_val in indicator_val.values():
            if "none" in case_val.keys():
                case_val.move_to_end("none", False)
            if "bnn" in case_val.keys():
                case_val.move_to_end("bnn", True)

    return results_npy

def get_color(net_type):
    if net_type == "bnn":
        return "red"
    elif net_type == "dnn":
        return "blue"
    else:
        return "black"

def get_text(key):
    # About Legend
    if key == "none":
        return legend_list[0]
    elif key == "dnn":
        return legend_list[1]
    elif key == "bnn":
        return legend_list[2]

    # About X-label
    elif key == "disturb":
        return xlabel_list[0]
    elif key == "uncertain":
        return xlabel_list[1]
    elif key == "noise":
        return xlabel_list[2]

    # About Y-label
    elif key == "reward":
        return ylabel_list[0]
    elif key == "success_rate":
        return ylabel_list[1]
    elif key == "model_error":
        return ylabel_list[2]

def plot_variance(plt, data, color, label):

    x = range(len(data[:, 0]))
    plt.plot(x, data[:, 0], 'o-', color=color, label=label)
    y1 = np.asarray(data[:, 0]) + np.asarray(data[:, 1])
    y2 = np.asarray(data[:, 0]) - np.asarray(data[:, 1])
    plt.fill_between(x, y1, y2, alpha=0.2, color=color)

def get_rewards_plot(plt, data):

    for net_type in data.keys():
        plot_variance(plt, data[net_type], color=get_color(net_type), label=get_text(net_type))

def get_success_rate_plot(plt, data):
    global num_data

    width = 0.2
    num_net_type = len(data.keys())
    for i, net_type in enumerate(data.keys()):

        x = np.arange(num_data)
        x_new = x + (i-(num_net_type-1)/2)*width
        rects = plt.bar(x_new, data[net_type].flatten(), width, color=get_color(net_type), label=get_text(net_type))
        # plt.bar_label(rects, padding=3)

    # plt.tight_layout()

def get_results_plot(data, case, min_case, max_case):
    global num_data

    width = 0.2
    num_net_type = 3

    fig, ax1 = plt.subplots(figsize=args.window_size)
    ax2 = ax1.twinx()

    reward_data = data["reward"][case]
    success_data = data["success_rate"][case]
    x = np.arange(num_data)  # the label locations

    for i, net_type in enumerate(reward_data.keys()):

        ax1.plot(x, reward_data[net_type][:, 0], 'o-', color=get_color(net_type), label=get_text(net_type))
        y1 = np.asarray(reward_data[net_type][:, 0]) + np.asarray(reward_data[net_type][:, 1])
        y2 = np.asarray(reward_data[net_type][:, 0]) - np.asarray(reward_data[net_type][:, 1])
        ax1.fill_between(x, y1, y2, alpha=0.1, color=get_color(net_type))

        x_new = x + (i-(num_net_type-1)/2)*width
        ax2.bar(x_new, success_data[net_type].flatten(), width, alpha=0.55, color=get_color(net_type), label=get_text(net_type))

    ax2.set_xticks(x, np.round(np.linspace(min_case, max_case, num_data), 2))
    ax1.set_xlabel(get_text(case), fontsize=args.label_size, fontweight='bold')
    ax1.set_ylabel(get_text("reward"), fontsize=args.label_size, fontweight='bold')
    ax2.set_ylabel(get_text("success_rate"), fontsize=args.label_size, fontweight='bold')
    ax2.legend(fontsize=args.legend_size, prop=dict(weight='bold'))

def get_selected_data(original_data):

    new_data = deepcopy(original_data)
    new = new_data["reward"]["disturb"]["dnn"][:, 0]
    new_data["reward"]["disturb"]["dnn"][:, 0] = new[new % 2 == 0]

    return new_data

def main():
    global num_data

    results_list = get_env_file(os.listdir("./results"))
    result_data = find_each_results(results_list)

    num_data = len(result_data["reward"]["disturb"]["dnn"])

    # result_data["model_error"]["uncertain"]["dnn"][:, 0] = result_data["model_error"]["uncertain"]["dnn"][:, 0] - np.min(result_data["model_error"]["uncertain"]["dnn"][:, 0])
    # result_data["model_error"]["uncertain"]["dnn"][:, 1] = result_data["model_error"]["uncertain"]["dnn"][:, 1] - np.min(result_data["model_error"]["uncertain"]["dnn"][:, 1])
    # result_data["model_error"]["uncertain"]["bnn"][:, 0] = result_data["model_error"]["uncertain"]["bnn"][:, 0] - np.min(result_data["model_error"]["uncertain"]["bnn"][:, 0])
    # result_data["model_error"]["uncertain"]["bnn"][:, 1] = result_data["model_error"]["uncertain"]["bnn"][:, 1] - np.min(result_data["model_error"]["uncertain"]["bnn"][:, 1])

    for i, indicator in enumerate(result_data.keys()):
        for j, case in enumerate(result_data[indicator].keys()):

            if (indicator == "model_error" and case != "noise") or \
                    (indicator != "model_error" and case == "noise"):
                continue

            if case == "disturb":
                min_case = args.min_disturb
                max_case = args.max_disturb
            elif case == "uncertain":
                min_case = -args.max_uncertain
                max_case = args.max_uncertain
            else:
                min_case = args.min_noise
                max_case = args.max_noise

            if args.each_plot:
                plt.figure()
                plt.xticks(np.arange(num_data), np.round(np.linspace(min_case, max_case, num_data), 2))

                if indicator == "success_rate":
                    get_success_rate_plot(plt, result_data[indicator][case])

                if indicator != "success_rate":
                    get_rewards_plot(plt, result_data[indicator][case])

                plt.xlabel(get_text(case), fontsize=args.label_size, fontweight='bold')
                plt.ylabel(get_text(indicator), fontsize=args.label_size, fontweight='bold')
                plt.legend(fontsize=args.legend_size, prop=dict(weight='bold'))
            else:
                if indicator == "model_error" and case == "noise":
                    plt.figure(figsize=args.window_size)
                    plt.xticks(np.arange(num_data), np.round(np.linspace(min_case, max_case, num_data), 2))
                    get_rewards_plot(plt, result_data[indicator][case])

                    plt.xlabel(get_text(case), fontsize=args.label_size, fontweight='bold')
                    plt.ylabel(get_text(indicator), fontsize=args.label_size, fontweight='bold')
                    plt.legend(fontsize=args.legend_size, prop=dict(weight='bold'))
                elif indicator == "reward" and case != "noise":
                    get_results_plot(result_data, case, min_case, max_case)
                else:
                    continue
    plt.show()

if __name__ == '__main__':
    main()
