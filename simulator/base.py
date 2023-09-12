from abc import abstractmethod
# abc (abstract base classes) library
# abstractmethod used so that the child classes that inherit the parent's must implement the function that has @abstractmethod

class Simulator():
    @abstractmethod
    def init_state(self, pose):
        return NotImplementedError

    @abstractmethod
    def step(self):
        return NotImplementedError

    @abstractmethod
    def render(self, img):
        return NotImplementedError