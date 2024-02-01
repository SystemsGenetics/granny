from GRANNY.Interfaces.UI.GrannyCLI import GrannyCLI
from GRANNY.Interfaces.UI.GrannyPyQt import GrannyPyQt


class GrannyBase(object):
    def __init__(self):
        self.granny_version: str = "1.0"

    def cli(self):
        GrannyCLI().run()

    def gui(self):
        GrannyPyQt().run()


def cli():
    GrannyBase().cli()


def gui():
    GrannyBase().gui()
