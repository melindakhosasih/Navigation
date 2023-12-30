import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import model.model as model

from algo.ddpg import DDPG
from algo.sac import SAC
from eval import run
from wrapper import NavEnv

def train(env, model_path, episode=1001, batch_size=64, eval_eps=50):
    total_step = 0
    max_success_rate = 0
    success_count = 0
    total_succ_rate = []
    overall_succ_rate = []
    succ_rate_split = []
    for eps in range(episode):
        state = env.initialize()
        step = 0
        loss_a = loss_c = 0
        total_reward = 0.
        
        while True:
            # Choose action
            action = env.model.choose_action(state, eval=False)

            # Step
            state_next, reward, done = env.step(env.translate_action(action))

            # Store
            end = 0 if done else 1
            env.model.store_transition(state, action, reward, state_next, end)

            # Render
            # self.render(gui=False)

            # Learn
            loss_a = loss_c = 0.
            if total_step > batch_size:
                loss_a, loss_c = env.model.learn()

            step += 1
            total_step += 1
            total_reward += reward
            if env.algo == "ddpg":
                print(f"\rEps:{eps:3d} /{step:4d} /{total_step:6d}| "
                    f"action_v:{action[0]:+.2f}| action_w:{action[1]:+.2f}| "
                    f"R:{reward:+.2f}| "
                    f"Loss:[A>{loss_a:+.2f} C>{loss_c:+.2f}]| "
                    f"Epsilon: {env.model.epsilon:.3f}| "
                    f"Ravg:{total_reward/step:.2f}", end='')
            elif env.algo == "sac":
                print(f"\rEps:{eps:3d} /{step:4d} /{total_step:6d}| "
                    f"action_v:{action[0]:+.2f}| action_w:{action[1]:+.2f}| "
                    f"R:{reward:+.2f}| "
                    f"Loss:[A>{loss_a:+.2f} C>{loss_c:+.2f}]| "
                    f"Alpha: {env.model.alpha:.3f}| "
                    f"Ravg:{total_reward/step:.2f}", end='')
            else:
                assert env.algo is None, "Algorithm doesn't exist"

            state = state_next.copy()
            if done or step>100:
                # Count the successful times
                if reward > 5:
                    success_count += 1
                    total_succ_rate.append(1)
                else:
                    total_succ_rate.append(0)
                print()
                break
        
        overall_succ_rate.append(np.mean(total_succ_rate))
        succ_rate_split.append(np.mean(total_succ_rate[-eval_eps:]))

        plot_fig(overall_succ_rate, succ_rate_split, model_path, eval_eps)

        if eps>0 and eps%eval_eps==0:
            # Sucess rate
            success_rate = success_count / eval_eps
            success_count = 0

            # Save the best model
            if success_rate >= max_success_rate:
                max_success_rate = success_rate
                print("Save model to " + model_path)
                env.model.save_load_model("save", model_path)
            print(f"Success Rate (current/max): {success_rate}/{max_success_rate}")
            # output GIF
            run(env.model, total_eps=4, gif_path=model_path+"/gif/", gif_name=f"{env.algo}_{str(eps).zfill(4)}.gif", message=True)

def plot_fig(overall_succ_rate, succ_rate_split, model_path, eval_eps):
    plt.plot(overall_succ_rate, label="Overall Training Succ")
    plt.plot(succ_rate_split, label=f"Avg Succ of Last {eval_eps} Episodes", linestyle='--')

    plt.xlabel('Episode')
    plt.ylabel('Succ')
    plt.legend()

    plt.savefig(f'{model_path}/training.png')
    plt.close()

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, required=True, help="Experiment's Title")
    parser.add_argument("--algo", default="ddpg", type=str, help="RL Algorithm")
    parser.add_argument("--map_path", default=None, type=str, help="Map's path")
    parser.add_argument("--epi", default=501, type=int, help="Number of Episodes")
    parser.add_argument("--batch", default=64, type=int, help="Batch Size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg()

    out_path = f"./out/{args.algo}/{args.title}/"
    map = None
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if args.map_path is not None:
        map = cv2.imread(args.map_path)

    if args.algo == "ddpg":
        model = DDPG(
            model = [model.PolicyNet, model.QNet],
                learning_rate = [0.0001, 0.0001],
                reward_decay = 0.99,
                memory_size = 10000,
                batch_size = 64
        )
    elif args.algo == "sac":
        model = SAC(
            model = [model.PolicyNetGaussian, model.QNet],
            n_actions = 2,
            learning_rate = [0.0001, 0.0001],
            reward_decay = 0.99,
            memory_size = 10000,
            batch_size = 64,
            alpha = 0.1,
            auto_entropy_tuning=True
        )
    else:
        assert args.algo is None, "Algorithm doesn't exist"

    env = NavEnv(model=model, map=map, algo=args.algo)
    train(env, model_path=out_path, episode=args.epi, batch_size=args.batch, eval_eps=50)
    print("Finished!!!")