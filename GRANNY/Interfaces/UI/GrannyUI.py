from abc import ABC, abstractmethod
from argparse import ArgumentParser


class GrannyUI(ABC):
    def __init__(self, parser: ArgumentParser):
        """
        Initalizes the GrannUI object.

        @param ArgumentParser parser
          Used to add arguments that this UI will need for calling
          granny on the command-line.
        """
        self.parser = parser

    @abstractmethod
    def run(self):
        """ """
        pass
