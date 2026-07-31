from argparse import ArgumentParser

from Granny.Interfaces.UI.GrannyUI import GrannyUI


class GrannyPyQt(GrannyUI):
    def __init__(self, parser: ArgumentParser):
        """
        {@inheritdoc}
        """
        GrannyUI.__init__(self, parser)

    def addProgramArgs(self):
        pass

    def run(self):
        import subprocess
        import sys
        import os
        gui_path = os.path.expanduser('~/softwares/granny-gui/granny_gui.py')
        python = sys.executable
        try:
            subprocess.run([python, gui_path])
        except KeyboardInterrupt:
            sys.exit(0)
