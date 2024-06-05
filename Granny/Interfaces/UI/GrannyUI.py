from abc import ABC, abstractmethod
from argparse import ArgumentParser


class GrannyUI(ABC):
    def __init__(self, parser: ArgumentParser):  # type:ignore
        """
        Initalizes the GrannUI object.

        @param ArgumentParser parser
          Used to add arguments that this UI will need for calling
          Granny on the command-line.
        """
        self.parser = parser

    @abstractmethod
    def configureParser(self, sub_parser):  # type: ignore
        """ """
        pass

    @abstractmethod
    def run(self):
        """ """
        pass
