from numpy.typing import NDArray


class MetaDataFile(MetaDataIO):
    def __init__(self, filepath: str):
        super(MetaDataIO, self).__init__()
        self.filepath: str = filepath
        self.metadata: NDArray = None

    def load():
        pass

    def save(metadata: MetaData):
        pass
