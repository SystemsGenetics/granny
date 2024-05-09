from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.backends import mps


class AIModel(ABC):
    """
    Class for loading, training, and saving Machine Learning instance segmentation
    model.
    """

    def __init__(self, model_dir: str):
        self.model_dir: str = model_dir
        self.model: Any
        self.device = (
            "cuda:0" if torch.cuda.is_available() else "mps" if mps.is_available() else "cpu"
        )

    @abstractmethod
    def loadModel(self):
        """
        Instantiates AI model for segmentation
        """
        pass

    @abstractmethod
    def getModel(self):
        """
        Gets segmentation model
        """
        pass
