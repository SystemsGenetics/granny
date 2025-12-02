"""
QR Code Detection Utility

This module provides functionality to detect and decode QR codes in images,
primarily used to extract variety information from tray images.

date: November 18, 2025
author: Aden Athar
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class QRCodeDetector:
    """
    Detects and decodes QR codes from images.

    This class uses OpenCV's QRCodeDetector to find QR codes in tray images
    and extract variety information (e.g., "BB-Late", "CC-Early").
    """

    def __init__(self):
        """Initialize the QR code detector."""
        self.detector = cv2.QRCodeDetector()

    def detect(self, image: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Detect and decode a QR code in an image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            Tuple of (decoded_data, points) where:
                - decoded_data: String containing QR code data, or None if not found
                - points: numpy array of QR code corner points, or None if not found
        """
        # Detect and decode QR code
        data, points, _ = self.detector.detectAndDecode(image)

        # Return data if found, otherwise None
        if data:
            return data, points
        return None, None

    def extract_variety_info(self, qr_data: str) -> dict:
        """
        Parse variety information from QR code data.

        Expected format: "BB-Late", "CC-Early", etc.

        Args:
            qr_data: Raw QR code string (e.g., "BB-Late")

        Returns:
            Dictionary with parsed variety information:
                {
                    'raw': 'BB-Late',      # Original QR code data
                    'full': 'BB-Late',     # Full variety string
                    'variety': 'BB',       # Variety code
                    'timing': 'Late'       # Timing info
                }
        """
        parts = qr_data.split('-')

        variety_info = {
            'raw': qr_data,
            'full': qr_data,
            'variety': parts[0] if len(parts) > 0 else '',
            'timing': parts[1] if len(parts) > 1 else ''
        }

        return variety_info
