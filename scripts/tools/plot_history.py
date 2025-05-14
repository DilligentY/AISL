from tensorboard.backend.event_processing import event_accumulator

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

parser = argparse.ArgumentParser(description="Visualization Reward History")
parser.add_argument("--library", type=str, default="skrl", help="Library for Reinforcement Learning")
parser.add_argument("--task", type=str, default="stack_franka", help="Name of the task.")
parser.add_argument("--start_env", type=int, default=0, help="Start environment index for plotting")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for plotting")
parser.add_argument("--max_step", type=int, default=36000, help="Max steps for plotting")
parser.add_argument("--plot_mode", type=str, default="mean", help="Plot mode: mean or merge")

args_cli, hydra_args = parser.parse_known_args()

logdir = os.path.join(os.getcwd(), "logs", args_cli.library, args_cli.task)
run_dirs = [d for d in glob.glob(os.path.join(logdir, '*')) if os.path.isdir(d)]

all_data = {}

for i in range(args_cli.start_env, args_cli.num_envs):
    run_dir = run_dirs[i]
    run_name = os.path.basename(run_dir)  # 예) run1, run2 등 폴더명
    ea = event_accumulator.EventAccumulator(run_dir)
    ea.Reload()
    
    # 이 폴더(실험)에 있는 모든 태그 정보 확인
    tags = ea.Tags()
    scalar_tags = tags.get('scalars', [])

    # 각 태그별로 스칼라 값을 가져와서 저장
    run_data = {}
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        values = [e.value for e in scalar_events]
        steps = [e.step for e in scalar_events]
        
        run_data[tag] = {
            'steps': steps,
            'values': values
        }
    
    # 모든 태그를 모아서 all_data에 저장
    all_data[run_name] = run_data

# 이제 all_data 딕셔너리에 각 실험(run)별, 태그별로 분류된 데이터가 담겨 있음
print(all_data.keys())  # dict_keys(['run1', 'run2', 'run3', ...])


def get_interp(values, max_step, steps, rew):
    # 데이터마다 길이가 다를 수 있으므로, 공통된 스텝을 만들어서 플로팅하기 위함.
    common_steps_pre = np.linspace(0, max_step, rew.shape[0])
    interpolated_values = np.interp(common_steps_pre, steps, values).reshape(-1, 1)
    rew = np.hstack([rew, interpolated_values])

    return rew, common_steps_pre

rew = None

if args_cli.plot_mode == "mean":

    for run_name, run_data in all_data.items():
        parts = run_name.split('_')
        part_hour = parts[1].split('-')[0]
        part_hour_int = int(part_hour)
        steps = run_data["Reward / Total reward (max)"]['steps']
        values = run_data["Reward / Total reward (max)"]['values']
        np_values = np.array(values).reshape(-1, 1)

        if rew is None:
            rew = np_values
        else:
            try:
                rew = np.hstack([rew, np_values])
            except:
                print("size mismatch")
                rew = get_interp(values, args_cli.max_step, steps, rew)

        plt.plot(steps, values, alpha=0.1, color="blue")

    mean_val = np.mean(rew, axis=1)

    plt.plot(steps, mean_val, color='red')
    plt.title('Reward History')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.show()

elif args_cli.plot_mode == "merge":
    total_steps = None

    for run_name, run_data in all_data.items():
        parts = run_name.split('_')
        part_hour = parts[1].split('-')[0]
        part_hour_int = int(part_hour)
        steps = run_data["Reward / Total reward (max)"]['steps']
        values = run_data["Reward / Total reward (max)"]['values']
        np_steps = np.array(steps).reshape(-1, 1)
        np_values = np.array(values).reshape(-1, 1)

        if rew is None:
            rew = np_values
            total_steps = np_steps
        else:
            try:
                rew = np.vstack([rew, np_values])
            except:
                print("size mismatch")
                rew, steps = get_interp(values, args_cli.max_step, steps, rew)

            np_steps += total_steps[-1]
            total_steps = np.vstack([total_steps, np_steps])


    plt.plot(total_steps, rew, color='red')
    plt.title('Reward History')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.show()


