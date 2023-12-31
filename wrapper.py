import cv2
import numpy as np
import argparse

from simulator.basic import BasicSimulator as Simulator
from simulator.utils import ControlState, Position, get_relative_pose

class NavEnv():
    def __init__(
            self,
            model=None,
            algo=None,
            map=None,
            dt=1,
            type="basic",
        ) -> None:
        self.map = np.asarray(map) if map is not None else np.ones((512, 512, 3))
        self.dt = dt
        self.sim_type = type
        self.model = model
        self.algo = algo
        self.prev_orien = 0.0

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

        reward_orien = self.prev_orien - np.deg2rad(abs(curr_deg))

        # Action Penalty
        reward_act = 0.05 if cmd[0] < -0.5 else 0
        # Total Reward
        # reward = 0.1*reward_dist - 0.2*reward_orien - reward_act
        reward = 0.1*reward_dist - 0.1*reward_orien

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
        # Update orientation
        self.prev_orien = np.deg2rad(abs(curr_deg))
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
       
    def _construct_state(self, relative_pose):
        # print("relative pose", relative_pose, end="\r")
        return [relative_pose[0]/10, np.cos(np.deg2rad(relative_pose[1])), np.sin(np.deg2rad(relative_pose[1]))]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default="0", help="0, 1")
    args = parser.parse_args()

    mode = ["random", "manual"]
    type = mode[args.mode]
    exit = False

    env = NavEnv()
    for i in range(5):
        if exit:
            break
        env.initialize()
        while True:
            key = cv2.waitKey(5)
            if key == 27: # ESC button
                print("exit")
                exit = True
                break
            elif key == ord("r") or key == ord("R"):
                # print("reset")
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