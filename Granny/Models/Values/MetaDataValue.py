import os
from typing import List

import pandas as pd
from Granny.Models.Images.Image import Image
from Granny.Models.Values.FileNameValue import FileNameValue
from Granny.Models.Values.Value import Value


class MetaDataValue(FileNameValue):
    """
    A value that stores the file name where the metadata are kept.

    The readValue() and writeValue() functions will read and write
    the metadata into the filename of the value.
    """

    def __init__(self, name: str, label: str, help: str):
        super().__init__(name, label, help)
        self.images: List[Image] = []

    def readValue(self):
        """ """
        pass

    def writeValue(self):
        """ """
        image_rating: pd.DataFrame = pd.DataFrame()
        for i, image_instance in enumerate(self.images):
            if i == 0:
                image_rating = pd.DataFrame(
                    columns=["Name"] + list(image_instance.getMetaData().keys())
                )
            output = [image_instance.getImageName()]
            for metadata in image_instance.getMetaData().values():
                output.append(metadata.getValue())
            image_rating.loc[i + 1] = output
        image_rating = image_rating.sort_values(by="Name").reset_index(drop=True)
        image_rating["TrayName"] = image_rating["Name"].str.extract(
            r"^(.*?)(?:_\d+)?\.(?:png|jpg|jpeg)$"
        )
        image_rating.to_csv(os.path.join(self.value, "results.csv"), header=True, index=False)
        tray_avg = image_rating.drop(columns=["Name"])
        tray_avg = tray_avg.groupby("TrayName").mean().reset_index()
        tray_avg.to_csv(os.path.join(self.value, "tray_summary.csv"), header=True, index=False)

    def getImageList(self):
        """ """
        return self.images

    def setImageList(self, images: List[Image]):
        """ """
        self.images = images
