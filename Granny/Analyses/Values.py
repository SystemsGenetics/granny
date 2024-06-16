import math
import os
from abc import ABC, abstractmethod
from typing import Any, List

from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile

# @todo: move this into the Models folder and separate the classes into
# individual files.


class Value(ABC):
    """
    The base abstract class for a values used by Granny.

    This class is used by all Analysis objects for representing the
    values that can be used to set how the analysis will perform.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        Instantiates a Value object.
        """
        self.name = name
        self.label = label
        self.help = help
        self.default_value = None
        self.type = Any
        self.value = None
        self.is_set = False

    @abstractmethod
    def validate(self) -> bool:
        """
        Validates the value matches the value constraints.
        """
        pass

    def getName(self) -> str:
        """
        Gets the machine readable name for this value
        """
        return self.name

    def getLabel(self) -> str:
        """
        Gets the human readable label for this value
        """
        return self.label

    def getHelp(self) -> str:
        return self.help

    def getType(self):
        """
        Gets the Python type for this value.
        """
        return self.type

    def setValue(self, value: Any):
        """
        Sets the current value of the value.
        """
        self.value = value
        self.is_set = True

    def isSet(self):
        """
        Indicates if the user set this value.

        If the value is not set then the getValue() function
        will return the default value.
        """
        return self.is_set

    def getValue(self) -> Any:
        """
        Returns the current value of the value.
        """
        if not self.is_set:
            return self.default_value
        else:
            return self.value

    def readValue(self):
        """
        Reads the value from the storage system.
        """
        pass

    def writeValue(self):
        """
        Writes the value to the storage system.
        """
        pass

    @abstractmethod
    def getDefaultValue(self) -> Any:
        """
        Gets the default value for the value.
        """
        pass

    @abstractmethod
    def setDefaultValue(self, value: Any):
        """
        Gets the default value for the value.
        """
        pass


class NumericValue(Value):
    """
    The base value class for a numeric value type
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = Any
        self.max_value = math.inf
        self.min_value = -math.inf

    @abstractmethod
    def setMax(self, value: Any):
        """
        Sets the maximum numeric value that this value can be set to.
        """
        pass

    @abstractmethod
    def setMin(self, value: Any):
        """
        Sets the minimum numeric value that this value can be set to.
        """
        pass


class BoolValue(Value):
    """
    Class for a boolean value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = bool

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True

    def getDefaultValue(self) -> bool:
        """
        {@inheritdoc}
        """
        return self.default_value

    def setDefaultValue(self, value: bool):
        """
        {@inheritdoc}
        """
        self.default_value = value


class IntValue(NumericValue):
    """
    Class for an integer value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = int
        self.value_values: list[int] = []
        self.value: int = 0

    def setMax(self, value: int):
        """
        {@inheritdoc}
        """
        self.max_value = value

    def setMin(self, value: int):
        """
        {@inheritdoc}
        """
        self.min_value = value

    def setDefaultValue(self, value: int):
        """
        Gets the default value for the value.
        """
        self.default_value = value

    def setValidValues(self, values: list[int]):
        """
        Provides a list of valid values for this integer value.
        """
        self.valid_values = values

    def getValidValues(self) -> List[int]:
        """
        Gets the list of valid values for this integer value.
        """
        return self.value_values

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True

    def getDefaultValue(self) -> int:
        """
        {@inheritdoc}
        """
        return self.default_value


class FloatValue(NumericValue):
    """
    Class for a float value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = float
        self.value_values: List[float] = []
        self.value: float = 0

    def setMax(self, value: float):
        """
        {@inheritdoc}
        """
        self.max_value = value

    def setMin(self, value: float):
        """
        {@inheritdoc}
        """
        self.min_value = value

    def setDefaultValue(self, value: float):
        """
        Sets the default value for the value.
        """
        self.default_value = value

    def getDefaultValue(self) -> float:
        """
        Gets the default value for the value.
        """
        return self.default_value

    def setValidValues(self, values: List[float]):
        """
        Provides a list of valid values for this integer value.
        """
        self.valid_values = values

    def getValidValues(self) -> List[float]:
        """
        Gets the list of valid values for this integer value.
        """
        return self.value_values

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True


class StringValue(Value):
    """
    Class for a string value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str
        self.value_values: List[str] = []

    def setValidValues(self, values: List[str]):
        """
        Provides a list of valid values for this string paratmer.
        """
        self.valid_values = values

    def getValidValues(self) -> List[str]:
        """
        Gets the list of valid values for this string value.
        """
        return self.value_values

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True

    def getDefaultValue(self) -> str:
        """
        {@inheritdoc}
        """
        return self.default_value

    def setDefaultValue(self, value: str):
        """
        {@inheritdoc}
        """
        self.default_value = value


class FileNameValue(Value):
    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str

    def validate(self) -> bool:
        """
        Makes sure that the filename is valid as either a file or a directory.
        """
        return True

    def getDefaultValue(self) -> str:
        """
        {@inheritdoc}
        """
        return self.default_value

    def setDefaultValue(self, value: str):
        """
        {@inheritdoc}
        """
        self.default_value = value


class FileDirValue(Value):
    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str

    def validate(self) -> bool:
        """
        Checks that the value provided is a valid directory on the file system

        @returns boolean
            returns True if the directory is valid, False otherwise.
        """
        return True


class ImageListValue(FileDirValue):
    """
    A value that stores the directory where images are kept.

    The readValue() and writeValue() functions will read and write
    the images that are in the directory provided as the value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = List[Image]
        self.images: List[Image] = []

        self.IMAGE_EXTENSION = (
            ".JPG",
            ".JPG".lower(),
            ".PNG",
            ".PNG".lower(),
            ".JPEG",
            ".JPEG".lower(),
            ".TIFF",
            ".TIFF".lower(),
        )

    def readValue(self, folder: str):
        """ """
        # reads image files from the input directory
        image_files: List[str] = os.listdir(self.value)  # type: ignore
        images = []

        for image_file in image_files:
            if image_file.endswith(self.IMAGE_EXTENSION):
                rgb_image = RGBImage(os.path.join(self.value, image_file))
                images.append(rgb_image)

        self.images = images

    def writeValue(self):
        """ """
        image_io: ImageIO = RGBImageFile()
        for image in self.images:
            image.saveImage(image_io)
        # @todo write the metadata for the image that
        # lives along side of the image.


class MetaDataValue(FileNameValue):
    """
    A value that stores the file name where the metadata are kept.

    The readValue() and writeValue() functions will read and write
    the metadata into the filename of the value.
    """

    def __init__(self, name: str, label: str, help: str):
        super().__init__(name, label, help)

    def readValue(self):
        """ """
        pass

    def writeValue(self):
        """ """
        pass
