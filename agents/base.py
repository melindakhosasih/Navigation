from abc import abstractmethod
from simulator.utils import State, ControlState

class AgentModel():
    @abstractmethod
    def step(self, state:State, cstate:ControlState) -> State:
        return NotImplementedError