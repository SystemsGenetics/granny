import cv2
import numpy as np
from Granny.Utils.QRCodeDetector import QRCodeDetector


def test_instantiation():
    detector = QRCodeDetector()
    assert detector.detector is not None
    assert isinstance(detector.barcode_enabled, bool)


def test_detect_returns_none_for_blank_image():
    detector = QRCodeDetector()
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    data, points = detector.detect(blank)
    assert data is None
    assert points is None


def test_detect_barcode_rotation_invariant():
    """Barcode detection should work regardless of image rotation."""
    detector = QRCodeDetector()
    if not detector.barcode_enabled:
        return

    # Create a test image with a Code128 barcode using pyzbar's expected input
    # We'll use a real barcode image if available, otherwise test the rotation logic
    # by generating a simple barcode-like pattern
    from pyzbar import pyzbar

    # Create a synthetic barcode image using python-barcode if available
    try:
        import barcode
        from barcode.writer import ImageWriter
        import tempfile
        import os

        code = barcode.get("code128", "TEST123", writer=ImageWriter())
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        code.save(tmp.name.replace(".png", ""))
        barcode_path = tmp.name.replace(".png", "") + ".png"

        img = cv2.imread(barcode_path)
        if img is None:
            return

        # Test original orientation
        data, points = detector._detect_barcode(img)
        assert data == "TEST123"

        # Test 90 degree rotation
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        data, points = detector._detect_barcode(rotated)
        assert data == "TEST123"

        # Test 180 degree rotation
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        data, points = detector._detect_barcode(rotated)
        assert data == "TEST123"

        # Test 270 degree rotation
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        data, points = detector._detect_barcode(rotated)
        assert data == "TEST123"

        os.unlink(barcode_path)
    except ImportError:
        # python-barcode not installed, skip
        pass


def test_extract_variety_info_pipe_format():
    detector = QRCodeDetector()
    info = detector.extract_variety_info("APPLE2026|LOT002|2026-01-23|BB-Early")
    assert info["project"] == "APPLE2026"
    assert info["lot"] == "LOT002"
    assert info["date"] == "2026-01-23"
    assert info["full"] == "BB-Early"
    assert info["variety"] == "BB"
    assert info["timing"] == "Early"


def test_extract_variety_info_legacy_format():
    detector = QRCodeDetector()
    info = detector.extract_variety_info("BB-Late")
    assert info["project"] == "UNKNOWN"
    assert info["lot"] == "UNKNOWN"
    assert info["full"] == "BB-Late"
    assert info["variety"] == "BB"
    assert info["timing"] == "Late"


def test_extract_variety_info_malformed_pipe():
    detector = QRCodeDetector()
    info = detector.extract_variety_info("only|two")
    assert info["project"] == "UNKNOWN"
    assert info["full"] == "only|two"
