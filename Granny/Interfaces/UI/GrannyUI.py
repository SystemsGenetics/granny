from abc import ABC, abstractmethod
from argparse import ArgumentParser


class GrannyUI(ABC):
    def __init__(self, parser: ArgumentParser):
        """
        Initializes the GrannyUI object.

        @param ArgumentParser parser
          Used to add arguments that this UI will need for calling
          Granny on the command-line.
        """
        self.parser = parser

    @abstractmethod
    def run(self):
        """ """
        pass
