import cv2
from GRANNY import GRANNY_Base as granny


class GrannyPearBlush(granny.GrannyBase):
    def __init__(self, action: str, fname: str):
        super().__init__(action, fname)
        self.threshold = 0

    def trackbar_change(self, val):
        self.threshold = val

    def calculate_blush_region(self) -> None:
        return

    def calibrate_blush_region(self) -> None:
        # cv2.namedWindow("Calibration")
        # cv2.createTrackbar("a*", "Calibration", 0, 255, self.trackbar_change)
        # cv2.setTrackbarMax("a*", "Calibration", 255)
        # cv2.setTrackbarMin("a*", "Calibration", 0)
        pass

    def main(self):
        self.calibrate_blush_region()


GrannyPearBlush("", "").main()
