from abc import ABC


class GrannyUI(ABC):
    def __init__(self):
        pass


class GrannyCLI(GrannyUI):
    def __init__(self):
        super(self, GrannyUI).__init__()
        self.image_dir: str = None
        self.metadata_dir: str = None
        self.result_dir: str = None
        self.analysis: str = None

    def checkArgs(self):
        # from params in analyses
        pass

    def printHelp(self):
        # from params in analyses
        pass

    def run(self):
        pass


class GrannyBase(object):
    # move this later
    def __init__(self):
        self.granny_version: str = None
        self.interface = None
