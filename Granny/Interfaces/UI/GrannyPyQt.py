from argparse import ArgumentParser

from Granny.Interfaces.UI.GrannyUI import GrannyUI


class GrannyPyQt(GrannyUI):
    def __init__(self, parser: ArgumentParser):
        """
        {@inheritdoc}
        """
        GrannyUI.__init__(self, parser)

    def configureParser(self, sub_parser):  # type:ignore
        self.gui_parser = sub_parser.add_parser(
            "gui",
            help="Graphical User Interface",
        )

    def run(self):
        pass
