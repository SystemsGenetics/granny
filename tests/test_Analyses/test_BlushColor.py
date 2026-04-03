import numpy as np
import pytest
from Granny.Analyses.BlushColor import BlushColor


def test_BlushColorInstantiation():
    analysis = BlushColor()
    assert analysis is not None
    assert analysis.__analysis_name__ == "blush"


def test_BlushColorInputImages():
    analysis = BlushColor()
    analysis.input_images.setValue("demo/pear_images/full_masked_images")
    assert analysis.input_images.getValue() == "demo/pear_images/full_masked_images"


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

def test_default_threshold_is_148():
    assert BlushColor().threshold.getValue() == 148


def test_default_fruit_threshold_is_140():
    assert BlushColor().fruit_threshold.getValue() == 140


def test_default_blush_color():
    a = BlushColor()
    assert a.blush_color_r.getValue() == 150
    assert a.blush_color_g.getValue() == 55
    assert a.blush_color_b.getValue() == 50


def test_default_text_position():
    a = BlushColor()
    assert a.text_x.getValue() == 20
    assert a.text_y.getValue() == 50


def test_default_font_scale_is_1():
    assert BlushColor().font_scale.getValue() == pytest.approx(1.0)


def test_default_text_thickness_is_3():
    assert BlushColor().text_thickness.getValue() == 3


def test_threshold_boundary_values():
    a = BlushColor()
    a.threshold.setValue(0)
    assert a.threshold.getValue() == 0
    a.threshold.setValue(255)
    assert a.threshold.getValue() == 255


# ---------------------------------------------------------------------------
# _calculateBlush with synthetic images
# ---------------------------------------------------------------------------

def _make_bgr(r, g, b, size=50):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def test_calculate_blush_returns_ratio_between_0_and_1():
    a = BlushColor()
    img = _make_bgr(200, 100, 100)
    ratio, _ = a._calculateBlush(img)
    assert 0.0 <= ratio <= 1.0


def test_calculate_blush_output_shape_matches_input():
    a = BlushColor()
    img = _make_bgr(200, 100, 100)
    _, result = a._calculateBlush(img)
    assert result.shape == img.shape


def test_calculate_blush_threshold_affects_ratio():
    img = _make_bgr(200, 150, 100)
    low_a = BlushColor()
    low_a.threshold.setValue(50)
    high_a = BlushColor()
    high_a.threshold.setValue(200)
    low_ratio, _ = low_a._calculateBlush(img)
    high_ratio, _ = high_a._calculateBlush(img)
    assert low_ratio != high_ratio


def test_calculate_blush_all_black_image():
    a = BlushColor()
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    ratio, result = a._calculateBlush(img)
    assert result.shape == img.shape


def test_calculate_blush_returns_float():
    a = BlushColor()
    img = _make_bgr(180, 100, 80)
    ratio, _ = a._calculateBlush(img)
    assert isinstance(ratio, float)
