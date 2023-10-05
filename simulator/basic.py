import cv2
import scipy
import numpy as np

from simulator.base import Simulator
from simulator.utils import ControlState, State, paste_overlapping_image

from agents.basic import BasicAgent

class BasicSimulator(Simulator):
    def __init__(self,
                 v_range=25,
                 w_range=45,
                 dt=0.1):
        # Initialize Agent
        self.agent = BasicAgent(dt)

        # Initialize State
        self.state = State()
        self.init_state((0, 0, 0))

        # Agent Type
        self.control_type = "basic"
        self.cstate = ControlState(self.control_type, 0.0, 0.0)

        # Agent Speed limit
        self.v_range = v_range
        self.w_range = w_range


    def __str__(self) -> str:
        return self.state.__str__() + " " + self.cstate.__str__()

    def init_state(self, pose):
        self.state.update(x=pose[0], y=pose[1], rotation=pose[2])
        # self.max_len = 1000 # Trajectory history len to render in GUI
        self.history = []
        
        try:
            img = cv2.imread("./agents/visualization/512x512.png", cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (25, 25))
            self.agent.agent_visualize(img)
        except Exception as e:
            print(f"Error loading agent image: {e}")
        
        return self.state, {}
    
    def step(self, cmd, update_state=True):
        if cmd is not None:
            # self.cstate.v = cmd.v if cmd.v is not None else self.cstate.v
            self.cstate.v = cmd.v if cmd.v is not None else 0
            # self.cstate.v = cmd.v
            # self.cstate.w = cmd.w if cmd.w is not None else self.cstate.w
            self.cstate.w = cmd.w if cmd.w is not None else 0
            # self.cstate.w = cmd.w

        # Control Constraint
        if self.cstate.v > self.v_range:
            self.cstate.v = self.v_range
        elif self.cstate.v < -self.v_range:
            self.cstate.v = -self.v_range
        if self.cstate.w > self.w_range:
            self.cstate.w = self.w_range
        elif self.cstate.w < -self.w_range:
            self.cstate.w = -self.w_range

        state_next = self.agent.step(self.state, self.cstate)
        if update_state:
            self.state = state_next
            self.history.append((self.state.pos.x, self.state.pos.y, self.state.rotation))

        return state_next
        # return state_next, {}
    
    def render(self, map=None):
        if map is None:
            map = np.ones((512, 512, 3))

        # Draw Trajectory
        # start = 0 if len(self.history) < self.max_len else len(self.history) - self.max_len
        
        color = (0/255, 97/255, 255/255)
        for i in range(0, len(self.history)-1):
            cv2.line(map, (int(self.history[i][0]), int(self.history[i][1])), (int(self.history[i+1][0]), int(self.history[i+1][1])), color, 1)

        if self.agent.img is not None:
            rotated_agent = scipy.ndimage.interpolation.rotate(
                self.agent.img, self.state.rotation + 90
            )
            rotated_agent = np.fliplr(rotated_agent)
            map = paste_overlapping_image(map, rotated_agent, (int(self.state.pos.y), int(self.state.pos.x)))

        return map
        