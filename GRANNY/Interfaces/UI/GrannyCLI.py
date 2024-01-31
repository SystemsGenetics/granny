from GRANNY.Interfaces.UI.GrannyUI import GrannyUI


class GrannyCLI(GrannyUI):
    def __init__(self):
        GrannyUI.__init__(self)
        self.image_dir: str = ""
        self.metadata_dir: str = ""
        self.result_dir: str = ""
        self.analysis: str = ""

    def checkArgs(self):
        # from params in analyses
        pass

    def printHelp(self):
        # from params in analyses
        pass

    def run(self):
        pass
