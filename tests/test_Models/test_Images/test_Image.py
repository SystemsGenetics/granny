from Granny.Models.Images.Image import Image
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.Values.Value import Value
from numpy.typing import NDArray
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from Granny.Models.Values.StringValue import StringValue

# Abstract method 
class testImage(Image):
    def rotateImage(self):
        """
        {@inheritdoc}
        """
        self.image = cast(NDArray[np.uint8], cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE))

    def getImage(self) -> NDArray[np.uint8]:
        """
        {@inheritdoc}
        """
        return self.image

    def setImage(self, image: NDArray[np.uint8]):
        """
        {@inheritdoc}
        """
        self.image = image

    def loadImage(self, image_io: ImageIO):
        """
        {@inheritdoc}
        """
        self.image = image_io.loadImage()

    def saveImage(self, image_io: ImageIO, folder: str):
        """
        {@inheritdoc}
        """
        image_io.saveImage(self.image, folder)
    def setMetaData(self, metadata: Dict[str, Value]):
        """
        {@inheritdoc}
        """
        for value in metadata.values():
            self.metadata[value.getName()] = value

    def getMetaData(self) -> Dict[str, Value]:
        """
        {@inheritdoc}
        """
        return self.metadata

    def setSegmentationResults(self, results: Any):
        """
        {@inheritdoc}
        """
        self.results = results

    def getSegmentationResults(self) -> Any:
        """
        {@inheritdoc}
        """
        return self.result

def test_add_getValue_getFilePath():
    filepath = "test-assets/images/single/cross_section_demo_image.jpeg"
    image = testImage(filepath)
    a = StringValue("name1", "label1", "help1")
    b = StringValue("name2", "label2", "help2")
    c = StringValue("name3", "label3", "help3")
    a.setValue('a-val')
    b.setValue('b-val')
    c.setValue('c-val')

    #val = [a,b,c]
    image.addValue(a, b, c)
    assert image.getValue('name1').getValue() == 'a-val'
    assert image.getFilePath() == os.path.abspath(filepath)
