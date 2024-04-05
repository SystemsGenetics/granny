from abc import ABC, abstractmethod
from pathlib import Path


class ImageIO(ABC):
    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.image_dir = Path(filepath).parent
        self.image_name = Path(filepath).name

    @abstractmethod
    def saveImage(self, folder: str):
        pass

    @abstractmethod
    def loadImage(self):
        pass

    @abstractmethod
    def getType(self) -> str:
        """
        Gets image type: RGB, Gray, Hyperspectral, ...
        """
        pass
