from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.backends import mps


class AIModel(ABC):
    """
    Abstract base class for loading, training, and managing an instance segmentation AI model.

    Handles model directory and device selection (CPU, CUDA, or MPS).
    """

    def __init__(self, model_dir: str):
        """
        Initializes the AI model with a given model directory and auto-selects the best available device.

        Args:
            model_dir (str): Path to the directory containing model weights/config.
        """
        self.model_dir: str = model_dir
        self.model: Any
        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps" if mps.is_available() else "cpu"
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
