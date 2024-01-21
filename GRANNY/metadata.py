class MetaData(object):
    def __init__(self):
        None


class MetaDataIO(object):
    def __init__(self):
        None


class MetaDataFile(MetaDataIO):
    def __init__(self):
        super(MetaDataIO, self).__init__()
        self.filepath: str = None
        self.metadata = None

    def load():
        pass

    def save(metadata: MetaData):
        pass
