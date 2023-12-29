import os
import cv2
import argparse

import model.model as model
from algo.ddpg import DDPG
from algo.sac import SAC

from wrapper import NavEnv

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, required=True, help="Experiment's Title")
    parser.add_argument("--algo", default="ddpg", type=str, help="RL Algorithm")
    parser.add_argument("--map_path", default=None, type=str, help="Map's path")
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
    env.train(model_path=out_path, eval_eps=50)
    print("Finished!!!")