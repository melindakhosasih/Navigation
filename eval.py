import os
import cv2
import argparse
import numpy as np

import model.model as model

from PIL import Image

from algo.ddpg import DDPG
from algo.sac import SAC
from wrapper import NavEnv

def run(model, gif_path, gif_name, total_eps=4, message=False):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    images = []
    env = NavEnv(model=model)
    for eps in range(total_eps):
        state = env.initialize()
        step = 0
        total_reward = 0.
        while True:
            # Choose action
            action = env.model.choose_action(state, eval=True)

            # Step
            state_next, reward, done = env.step(env.translate_action(action))

            # Render
            img = env.render(gui=False)
            img = Image.fromarray(cv2.cvtColor(np.uint8(img*255),cv2.COLOR_BGR2RGB))
            images.append(img)

            total_reward += reward

            if message:
                print(f"\rEps:{eps:2d} /{step:4d} | action:{action[0]:+.2f}| "
                        f"R:{reward:+.2f} | Total R:{total_reward:.2f}", end='')

            state = state_next.copy()
            step += 1

            if done or step>=100:
                # Count the successful times
                if message:
                    print()
                break

    print("Save evaluation GIF ...")
    if gif_path is not None:
        images[0].save(gif_path+gif_name,
            save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="results/", type=str, help="'s Title")
    parser.add_argument("--result_name", type=str, required=True, help="'s Title")
    parser.add_argument("--model_dir", type=str, required=True, help="Model's Directory")
    parser.add_argument("--algo", type=str, required=True, help="RL Algorithm")
    parser.add_argument("--map_path", default=None, type=str, help="Map's path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg()
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

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

    model.save_load_model("load", args.model_dir)
    run(model, total_eps=4, gif_path=args.result_dir, gif_name=f"{args.result_name}.gif", message=True)
