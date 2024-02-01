from abc import ABC, abstractmethod


class GrannyUI(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass
