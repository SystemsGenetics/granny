"""
Integration tests for analysis pipelines.

Unlike unit tests that call _calculateStarch / _calculateBlush directly with
in-memory numpy arrays, these tests exercise the full _processImage path:
write a real PNG to disk → create RGBImage from that path → call _preRun +
_processImage → verify the result has the expected structure and values.

This catches bugs in file I/O, Image/ImageIO wiring, and Value assembly that
unit tests miss (e.g. the FileDirValue validation-order bug documented in
BUGFIXES.md would have surfaced here).
"""

import os

import cv2
import numpy as np
import pytest

from Granny.Analyses.BlushColor import BlushColor
from Granny.Analyses.StarchArea import StarchArea, StarchScales
from Granny.Models.Images.RGBImage import RGBImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_png(path: str, bgr_value: tuple, size: int = 80) -> None:
    """Write a solid-colour BGR image to *path* as a PNG."""
    img = np.full((size, size, 3), bgr_value, dtype=np.uint8)
    cv2.imwrite(path, img)


def _write_mixed_starch_png(path: str, dark_fraction: float = 0.5, size: int = 100) -> None:
    """
    Write a two-tone image with a controlled dark/bright split.

    Uses values 20 (dark) and 200 (bright) so normalization is never degenerate.
    After normalizing to [0, 255], the dark pixels land near 0 and the bright
    ones near 255, giving predictable starch classification at threshold=140.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    split = int(size * dark_fraction)
    img[:, :split] = 20     # dark → starch after normalisation
    img[:, split:] = 200    # bright → no starch
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# StarchArea integration
# ---------------------------------------------------------------------------

class TestStarchAreaIntegration:
    def test_processimage_returns_rgb_image(self, tmp_path):
        """_processImage must return an Image with the underlying array set."""
        img_path = str(tmp_path / "apple_fruit_01.png")
        _write_synthetic_png(img_path, bgr_value=(30, 30, 30))

        analysis = StarchArea()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        assert result is not None
        assert result.getImage() is not None

    def test_processimage_result_has_rating_value(self, tmp_path):
        """The result Image must expose a 'rating' FloatValue after processing."""
        img_path = str(tmp_path / "apple_fruit_01.png")
        _write_synthetic_png(img_path, bgr_value=(80, 80, 80))

        analysis = StarchArea()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        assert "rating" in result.getMetaData(), "result is missing 'rating' key"
        rating = result.getValue("rating").getValue()
        assert 0.0 <= rating <= 1.0, f"rating {rating} is out of [0, 1] range"

    def test_processimage_result_has_all_scale_indices(self, tmp_path):
        """The result must include a starch index for every variety in StarchScales."""
        img_path = str(tmp_path / "apple_fruit_01.png")
        _write_mixed_starch_png(img_path)

        analysis = StarchArea()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        varieties = {k for k in vars(StarchScales) if not k.startswith("_")}
        metadata_keys = set(result.getMetaData().keys())
        missing = varieties - metadata_keys
        assert not missing, f"Missing starch scale indices: {missing}"

    def test_processimage_higher_threshold_gives_higher_rating(self, tmp_path):
        """Higher threshold should classify more pixels as starch on the same image."""
        img_path = str(tmp_path / "apple_fruit_01.png")
        _write_mixed_starch_png(img_path, dark_fraction=0.5)

        low_analysis = StarchArea()
        low_analysis.starch_threshold.setValue(80)
        low_analysis._preRun()
        low_result = low_analysis._processImage(RGBImage(img_path))

        high_analysis = StarchArea()
        high_analysis.starch_threshold.setValue(200)
        high_analysis._preRun()
        high_result = high_analysis._processImage(RGBImage(img_path))

        low_rating = low_result.getValue("rating").getValue()
        high_rating = high_result.getValue("rating").getValue()
        assert high_rating >= low_rating, (
            f"Higher threshold should give higher starch ratio: low={low_rating}, high={high_rating}"
        )

    def test_processimage_result_image_same_shape_as_input(self, tmp_path):
        """The annotated output image must have the same spatial dimensions as the input."""
        img_path = str(tmp_path / "apple_fruit_01.png")
        size = 64
        _write_synthetic_png(img_path, bgr_value=(100, 100, 100), size=size)

        analysis = StarchArea()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        h, w, _ = result.getImage().shape
        assert (h, w) == (size, size)

    def test_processimage_preserves_image_name(self, tmp_path):
        """getImageName() on the result should match the input filename."""
        img_path = str(tmp_path / "apple_fruit_01.png")
        _write_synthetic_png(img_path, bgr_value=(128, 128, 128))

        analysis = StarchArea()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        assert result.getImageName() == "apple_fruit_01.png"


# ---------------------------------------------------------------------------
# BlushColor integration
# ---------------------------------------------------------------------------

class TestBlushColorIntegration:
    def _yellow_pear_image(self, path: str, size: int = 80) -> None:
        """Write a yellow pear image: high B channel (fruit) with some A (blush)."""
        # In LAB (OpenCV 0–255 encoding): L~128, A~148 (neutral), B~200 (yellow)
        # We create a BGR image that converts to a known LAB range.
        # A yellow-ish image: R≈200, G≈180, B≈50 in BGR → yellow pear background
        img = np.full((size, size, 3), (50, 180, 200), dtype=np.uint8)
        cv2.imwrite(path, img)

    def _blush_pear_image(self, path: str, size: int = 80) -> None:
        """Write an image where ~half the pixels will register as blush (high A channel)."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Left half: reddish (will have high A in LAB) — BGR (0, 80, 200)
        img[:, :size // 2] = (0, 80, 200)
        # Right half: yellow (low A, high B) — BGR (50, 180, 200)
        img[:, size // 2:] = (50, 180, 200)
        cv2.imwrite(path, img)

    def test_processimage_returns_image(self, tmp_path):
        img_path = str(tmp_path / "pear_fruit_01.png")
        self._yellow_pear_image(img_path)

        analysis = BlushColor()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        assert result is not None
        assert result.getImage() is not None

    def test_processimage_result_has_rating_value(self, tmp_path):
        img_path = str(tmp_path / "pear_fruit_01.png")
        self._yellow_pear_image(img_path)

        analysis = BlushColor()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        assert "rating" in result.getMetaData()
        rating = result.getValue("rating").getValue()
        assert 0.0 <= rating <= 1.0, f"rating {rating} out of range"

    def test_processimage_result_image_same_shape(self, tmp_path):
        size = 60
        img_path = str(tmp_path / "pear_fruit_01.png")
        img = np.full((size, size, 3), (50, 180, 200), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        analysis = BlushColor()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        h, w, _ = result.getImage().shape
        assert (h, w) == (size, size)

    def test_processimage_preserves_image_name(self, tmp_path):
        img_path = str(tmp_path / "pear_fruit_01.png")
        self._yellow_pear_image(img_path)

        analysis = BlushColor()
        analysis._preRun()
        result = analysis._processImage(RGBImage(img_path))

        assert result.getImageName() == "pear_fruit_01.png"

    def test_processimage_higher_blush_image_gives_higher_rating(self, tmp_path):
        """An image with more reddish pixels should produce a higher blush rating."""
        low_path = str(tmp_path / "low_blush_fruit_01.png")
        high_path = str(tmp_path / "high_blush_fruit_01.png")
        self._yellow_pear_image(low_path)
        self._blush_pear_image(high_path)

        analysis = BlushColor()
        analysis._preRun()
        low_result = analysis._processImage(RGBImage(low_path))
        high_result = analysis._processImage(RGBImage(high_path))

        low_rating = low_result.getValue("rating").getValue()
        high_rating = high_result.getValue("rating").getValue()
        assert high_rating >= low_rating, (
            f"Expected blush image to score higher: low={low_rating}, high={high_rating}"
        )
