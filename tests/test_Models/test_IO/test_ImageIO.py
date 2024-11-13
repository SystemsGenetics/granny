from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.IO.ImageIO import ImageIO
from numpy.typing import NDArray
import numpy as np

# Abstract Method 
# I copied the code for the methods in the test class from RGBImageFile since
# we are not testing if these methods work in this file. We are only testing
# the setFilePath method as it is not an abstract method
class testImageIO(ImageIO):
    def loadImage(self) -> NDArray[np.uint8]:
        """
        {@inheritdoc}
        """
        # loads image in the BGR format (default to OpenCV)
        image = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
        return cast(NDArray[np.uint8], image)

    def saveImage(self, image: NDArray[np.uint8], output_path: str) -> None:
        """
        {@inheritdoc}
        """
        if not os.path.exists(os.path.join(output_path)):
            os.makedirs(os.path.join(output_path), exist_ok=True)
        cv2.imwrite(os.path.join(output_path, self.image_name), image)

    def getType(self):
        """
        {@inheritdoc}
        """
        return RGBImageFile.__image_type__


def test_setFilePath():
    filePath = "demo/cross_section_images/cross_section_tray/cross_section_demo_image.jpeg"
    image = testImageIO()
    image.setFilePath(filePath)
    assert image.filepath == filePath