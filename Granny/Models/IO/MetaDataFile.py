from Granny.Models.Images.MetaData import MetaData
from Granny.Models.IO.MetaDataIO import MetaDataIO


class MetaDataFile(MetaDataIO):
    def __init__(self, filepath: str):
        MetaDataIO.__init__(self, filepath)
        self.metadata: MetaData

    def load(self):
        pass

    def save(self, metadata: MetaData):
        pass
