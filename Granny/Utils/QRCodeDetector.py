"""
QR Code and Barcode Detection Utility

This module provides functionality to detect and decode QR codes and barcodes
in images, primarily used to extract variety information from tray images.

date: November 18, 2025
author: Aden Athar
"""

import cv2
import numpy as np
from typing import Optional, Tuple

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError as e:
    PYZBAR_AVAILABLE = False
    PYZBAR_ERROR = str(e)


class QRCodeDetector:
    """
    Detects and decodes QR codes and barcodes from images.

    This class uses OpenCV's QRCodeDetector for QR codes and pyzbar for
    1D barcodes (Code128, Code39, EAN, UPC, etc.) to find codes in tray
    images and extract variety information (e.g., "BB-Late", "CC-Early").
    """

    def __init__(self):
        """Initialize the QR code and barcode detector."""
        self.detector = cv2.QRCodeDetector()
        self.barcode_enabled = PYZBAR_AVAILABLE

        if not PYZBAR_AVAILABLE:
            print("WARNING: Barcode detection unavailable. Install libzbar0:")
            print("  Ubuntu/Debian: sudo apt-get install libzbar0")
            print("  macOS: brew install zbar")
            print("  Windows: Download from http://zbar.sourceforge.net/")

    def detect(self, image: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Detect and decode a QR code or barcode in an image.

        Tries QR code detection first, then falls back to barcode detection
        if no QR code is found and pyzbar is available.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            Tuple of (decoded_data, points) where:
                - decoded_data: String containing code data, or None if not found
                - points: numpy array of code corner points, or None if not found
        """
        # Try QR code detection first
        data, points, _ = self.detector.detectAndDecode(image)

        if data:
            return data, points

        # Fall back to barcode detection if pyzbar is available
        if self.barcode_enabled:
            barcode_data, barcode_points = self._detect_barcode(image)
            if barcode_data:
                return barcode_data, barcode_points

        return None, None

    def _detect_barcode(self, image: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Detect and decode a barcode using pyzbar, trying multiple rotations.

        Barcodes may appear at any angle in the image. This method tries the
        original orientation first, then rotates by 90, 180, and 270 degrees
        to ensure detection regardless of how the image was captured.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            Tuple of (decoded_data, points) where:
                - decoded_data: String containing barcode data, or None if not found
                - points: numpy array of barcode corner points, or None if not found
        """
        if not PYZBAR_AVAILABLE:
            return None, None

        # Convert to grayscale for better detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Try original and 3 rotations (0, 90, 180, 270 degrees)
        rotations = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]

        for rotation in rotations:
            rotated = gray if rotation is None else cv2.rotate(gray, rotation)
            barcodes = pyzbar.decode(rotated)

            if barcodes:
                barcode = barcodes[0]
                data = barcode.data.decode('utf-8')
                points = np.array(barcode.polygon, dtype=np.float32)
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
