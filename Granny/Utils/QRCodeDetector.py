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

        Supports two formats:
        1. New format: "PROJECT|LOT|DATE|VARIETY" (pipe-delimited)
        2. Legacy format: "BB-Late" (dash-separated variety only)

        Args:
            qr_data: Raw QR code string

        Returns:
            Dictionary with parsed variety information:
                {
                    'raw': original QR string,
                    'project': project code or 'UNKNOWN',
                    'lot': lot code or 'UNKNOWN',
                    'date': date string or 'UNKNOWN',
                    'variety': variety code (e.g., 'BB'),
                    'timing': timing info (e.g., 'Late'),
                    'full': full variety string (e.g., 'BB-Late')
                }
        """
        variety_info = {'raw': qr_data}

        # Check if new pipe-delimited format
        if '|' in qr_data:
            parts = qr_data.split('|')
            if len(parts) >= 4:
                variety_info['project'] = parts[0]
                variety_info['lot'] = parts[1]
                variety_info['date'] = parts[2]
                variety_info['full'] = parts[3]

                # Parse variety and timing from full variety string
                variety_parts = parts[3].split('-')
                variety_info['variety'] = variety_parts[0] if len(variety_parts) > 0 else ''
                variety_info['timing'] = variety_parts[1] if len(variety_parts) > 1 else ''
            else:
                # Malformed pipe-delimited format
                variety_info.update({
                    'project': 'UNKNOWN',
                    'lot': 'UNKNOWN',
                    'date': 'UNKNOWN',
                    'full': qr_data,
                    'variety': '',
                    'timing': ''
                })
        else:
            # Legacy format (just variety, e.g., "BB-Late")
            parts = qr_data.split('-')
            variety_info.update({
                'project': 'UNKNOWN',
                'lot': 'UNKNOWN',
                'date': 'UNKNOWN',
                'full': qr_data,
                'variety': parts[0] if len(parts) > 0 else '',
                'timing': parts[1] if len(parts) > 1 else ''
            })

        return variety_info
