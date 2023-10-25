import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

from simulator.basic import BasicSimulator as Simulator
from simulator.utils import ControlState, Position, get_relative_pose

from PIL import Image

import model.model as model

class NavEnv():
    def __init__(
            self,
            model,
            algo,
            map=None,
            dt=1,
            type="basic",
        ) -> None:
        self.map = np.asarray(map) if map is not None else np.ones((512, 512, 3))
        self.dt = dt
        self.sim_type = type
        self.model = model
        self.algo = algo

    def initialize(self):
        # Initialize Agent Position
        self.env = Simulator(dt=self.dt)
        self.env.state.pos = self.random_position()
        self.env.state.rotation = 360 * np.random.random()
        self.goal_dist = 0
        relative_pose = []

        # while(self.goal_dist <= 1.5 or self.goal_dist > 3):
        while(self.goal_dist < 3 or self.goal_dist > 5):
            # Initialize Goal Position
            self.goal = self.random_position()

            # Compute Relative Pose
            relative_pose = get_relative_pose(self.env.state.pos, self.env.state.rotation, self.goal)
            self.goal_dist = relative_pose[0]

        # Get State
        state = self._construct_state(relative_pose)
        return state

    def random_position(self):
        h, w = self.map.shape[0], self.map.shape[1]
        x = np.random.randint(0, h-50)
        y = np.random.randint(0, w-50)
        return Position(x, y)

    def render(self, gui=False):
        img = self.map.copy()
        # Draw Target
        cv2.circle(img, (int(1*self.goal.x), int(1*self.goal.y)), 100, (0, 0, 1), 2)
        
        # Render Agemt and Trajectory
        self.env.render(img)
        
        if gui:
            cv2.imshow("demo", img)
            cv2.waitKey(1)

        return img.copy()
        
    def step(self, cmd):
        """
            Check state (Position) This is not RL's state:
            1. state : State() class
                - state.pos.x
                - state.pos.y
                - state.rotation
                - state.v
                - state.w
            2. self.env.state : State() class
                - self.env.state.pos.x
                - self.env.state.pos.y
                - self.env.state.rotation
                - self.env.state.v
                - self.env.state.w
            3. print(state)
            4. print(self.env)

            Relative Pose
            - relative_pose[0] : distance
            - relative_pose[1] : degree (in 360, not radian)
        """
        # Get next state (position info and velocity)
        # self.env.step(ControlState(self.sim_type, (cmd[0]+1)/2 * self.env.v_range, cmd[1] * self.env.w_range))
        self.env.step(ControlState(self.sim_type, cmd[0], cmd[1]))
        relative_pose = get_relative_pose(self.env.state.pos, self.env.state.rotation, self.goal)
        state_next = self._construct_state(relative_pose)

        # Compute reward
        curr_dist = relative_pose[0]
        curr_deg = relative_pose[1]
        # print(relative_pose, end="\r")
        # Distance Reward
        reward_dist = self.goal_dist - curr_dist
        reward_dist *= 100

        # Orientation Reward
        while curr_deg > 180:
            curr_deg -= 360
        while curr_deg < -180:
            curr_deg += 360

        reward_orien = np.deg2rad(abs(curr_deg))

        # Action Penalty
        reward_act = 0.05 if cmd[0] < -0.5 else 0
        # Total Reward
        reward = 0.1*reward_dist - 0.2*reward_orien - reward_act
        # reward = 0.1 * reward_dist - 0.1 * reward_orien

        # Check Boundary
        collision = False
        if self.env.state.pos.x < 0 or self.env.state.pos.y < 0 or \
            self.env.state.pos.x > self.map.shape[1] or self.env.state.pos.y > self.map.shape[0]:
            collision = True

        # Check if arrive at goal or out of boundary
        done = False
        if curr_dist < 1:
            reward = 20
            done = True
        # elif curr_dist > 5:
        #     reward += -0.1
        # elif collision:
        #     reward = -2
        #     done = True

        # Update distance
        self.goal_dist = curr_dist
        return state_next, reward, done
    
    def translate_action(self, action):
        velocity = (action[0] + 1) / 2
        forward = round(velocity / 0.01)
        
        turn = abs(round(action[1] / 0.1))

        turn_action = 1 * turn  # turn right

        if action[1] < 0:   # turn left
            turn_action *= -1

        forward_action = forward * 0.1
        return [forward_action, turn_action]
    
    def train(self, model_path, episode=1001, batch_size=64, eval_eps=50):
        total_step = 0
        max_success_rate = 0
        success_count = 0
        total_succ_rate = []
        overall_succ_rate = []
        succ_rate_split = []
        for eps in range(episode):
            state = self.initialize()
            step = 0
            loss_a = loss_c = 0
            total_reward = 0.
            
            while True:
                # Choose action
                action = self.model.choose_action(state, eval=False)

                # Step
                state_next, reward, done = self.step(self.translate_action(action))

                # Store
                end = 0 if done else 1
                self.model.store_transition(state, action, reward, state_next, end)

                # Render
                # self.render(gui=False)

                # Learn
                loss_a = loss_c = 0.
                if total_step > batch_size:
                    loss_a, loss_c = self.model.learn()

                step += 1
                total_step += 1
                total_reward += reward
                if self.algo == "ddpg":
                    print(f"\rEps:{eps:3d} /{step:4d} /{total_step:6d}| "
                      f"action_v:{action[0]:+.2f}| action_w:{action[1]:+.2f}| "
                      f"R:{reward:+.2f}| "
                      f"Loss:[A>{loss_a:+.2f} C>{loss_c:+.2f}]| "
                      f"Epsilon: {self.model.epsilon:.3f}| "
                      f"Ravg:{total_reward/step:.2f}", end='')
                elif self.algo == "sac":
                    print(f"\rEps:{eps:3d} /{step:4d} /{total_step:6d}| "
                      f"action_v:{action[0]:+.2f}| action_w:{action[1]:+.2f}| "
                      f"R:{reward:+.2f}| "
                      f"Loss:[A>{loss_a:+.2f} C>{loss_c:+.2f}]| "
                      f"Alpha: {self.model.alpha:.3f}| "
                      f"Ravg:{total_reward/step:.2f}", end='')
                else:
                    assert self.algo is None, "Algorithm doesn't exist"

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

            self.plot_fig(overall_succ_rate, succ_rate_split, model_path, eval_eps)

            if eps>0 and eps%eval_eps==0:
                # Sucess rate
                success_rate = success_count / eval_eps
                success_count = 0

                # Save the best model
                if success_rate >= max_success_rate:
                    max_success_rate = success_rate
                    print("Save model to " + model_path)
                    self.model.save_load_model("save", model_path)
                print(f"Success Rate (current/max): {success_rate}/{max_success_rate}")
                # output GIF
                self.eval(self.model, total_eps=4, gif_path=model_path+"/gif/", gif_name=f"{self.algo}_"+str(eps).zfill(4)+".gif", message=True)

    def eval(self, model, gif_path, gif_name, total_eps=4, message=False):
        if not os.path.exists(gif_path):
            os.makedirs(gif_path)

        images = []
        for eps in range(total_eps):
            state = self.initialize()
            step = 0
            total_reward = 0.
            while True:
                # Choose action
                action = self.model.choose_action(state, eval=False)

                # Step
                state_next, reward, done = self.step(self.translate_action(action))

                # Render
                img = self.render(gui=False)
                img = Image.fromarray(cv2.cvtColor(np.uint8(img*255),cv2.COLOR_BGR2RGB))
                images.append(img)

                total_reward += reward

                if message:
                    print(f"\rEps:{eps:2d} /{step:4d} | action:{action[0]:+.2f}| "
                          f"R:{reward:+.2f} | Total R:{total_reward:.2f}", end='')

                state = state_next.copy()
                step += 1

                if done or step>100:
                    # Count the successful times
                    if message:
                        print()
                    break

        print("Save evaluation GIF ...")
        if gif_path is not None:
            images[0].save(gif_path+gif_name,
                save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)

    # def _construct_state(self, relative_pose):
    #     state = relative_pose.copy()
    #     state[1] = np.deg2rad(state[1])
    #     state = [relative_pose[0]*np.cos(state[1]), relative_pose[0]*np.sin(state[1])]
    #     state[0] /= 5
    #     state[1] /= 5
    #     return state
    
    def _construct_state(self, relative_pose):
        # print("relative pose", relative_pose, end="\r")
        return [relative_pose[0]/10, np.cos(np.deg2rad(relative_pose[1])), np.sin(np.deg2rad(relative_pose[1]))]

    def plot_fig(self, overall_succ_rate, succ_rate_split, model_path, eval_eps):
        plt.plot(overall_succ_rate, label="Overall Training Succ")
        plt.plot(succ_rate_split, label=f"Avg of {eval_eps} Episodes", linestyle='--')

        plt.xlabel('Episode')
        plt.ylabel('Succ')
        plt.legend()

        plt.savefig(f'{model_path}/training.png')
        plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default="0", help="0, 1")
    args = parser.parse_args()

    mode = ["random", "manual"]
    type = mode[args.mode]

    env = NavEnv(model=None, algo=None)
    for i in range(5):
        env.initialize()
        while True:
            key = cv2.waitKey(5)
            if key == 27: # ESC button
                # print("exit")
                break
            if type == "random":
                action = 2*np.random.random(2) - 1
            elif type == "manual":
                if key == ord("w") or key == ord("W"):
                    # print("move forward")
                    action = [1, 0]
                elif key == ord("a") or key == ord("A"):
                    # print("turn left")
                    action = [-1, -1]
                elif key == ord("s") or key == ord("S"):
                    # print("move backward")
                    action = [-1, 0]
                elif key == ord("d") or key == ord("D"):
                    # print("turn right")
                    action = [-1, 1]
                else:
                    action = [-1, 0]

            state_next, reward, done = env.step(env.translate_action(action))
            # print the state after env.step
            # print(env.env, end="\r")
            # print(env.env)

            # print(state_next, reward, done, end="\r")
            env.render(gui=True)

            if done:
                break
    cv2.destroyAllWindows()