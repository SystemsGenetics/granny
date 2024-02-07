from argparse import ArgumentParser

from GRANNY.Interfaces.UI.GrannyUI import GrannyUI


class GrannyPyQt(GrannyUI):
    def __init__(self, parser: ArgumentParser):
        """
        {@inheritdoc}
        """
        GrannyUI.__init__(self, parser)

    def run(self):
        pass
