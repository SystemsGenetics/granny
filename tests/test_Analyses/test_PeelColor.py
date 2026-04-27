import numpy as np
import pytest
from Granny.Analyses.PeelColor import PeelColor


def test_PeelColorInstantiation():
    analysis = PeelColor()
    assert analysis is not None
    assert analysis.__analysis_name__ == "color"


def test_PeelColorInputImages():
    analysis = PeelColor()
    analysis.input_images.setValue("demo/pear_images/full_masked_images")
    assert analysis.input_images.getValue() == "demo/pear_images/full_masked_images"


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

def test_default_purple_threshold_is_126():
    assert PeelColor().purple_threshold.getValue() == 126


def test_default_lightness_range():
    a = PeelColor()
    assert a.lightness_min.getValue() == 0
    assert a.lightness_max.getValue() == 255


def test_default_green_range():
    a = PeelColor()
    assert a.green_min.getValue() == 0
    assert a.green_max.getValue() == 128


def test_default_yellow_range():
    a = PeelColor()
    assert a.yellow_min.getValue() == 128
    assert a.yellow_max.getValue() == 255


def test_default_normalize_lightness_is_50():
    assert PeelColor().normalize_lightness.getValue() == 50


def test_mean_values_length_match():
    a = PeelColor()
    assert len(a.MEAN_VALUES_A) == len(a.MEAN_VALUES_B) == len(a.SCORE)


def test_line_points_are_2d():
    a = PeelColor()
    assert a.LINE_POINT_1.shape == (2,)
    assert a.LINE_POINT_2.shape == (2,)


# ---------------------------------------------------------------------------
# remove_purple
# ---------------------------------------------------------------------------

def test_remove_purple_output_shape():
    a = PeelColor()
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    result = a.remove_purple(img)
    assert result.shape == img.shape


def test_remove_purple_does_not_modify_original():
    a = PeelColor()
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    original = img.copy()
    a.remove_purple(img)
    np.testing.assert_array_equal(img, original)


# ---------------------------------------------------------------------------
# calculate_bin_distance
# ---------------------------------------------------------------------------

def test_calculate_bin_distance_euclidean_returns_valid_bin():
    a = PeelColor()
    bin_num, dist = a.calculate_bin_distance([-20.0, 70.0], method="Euclidean")
    assert 1 <= bin_num <= len(a.SCORE)


def test_calculate_bin_distance_score_method():
    a = PeelColor()
    bin_num, dist = a.calculate_bin_distance([0.6], method="Score")
    assert 1 <= bin_num <= len(a.SCORE)


def test_calculate_bin_distance_x_component():
    a = PeelColor()
    bin_num, _ = a.calculate_bin_distance([-20.0, 70.0], method="X-component")
    assert 1 <= bin_num <= len(a.SCORE)


def test_calculate_bin_distance_y_component():
    a = PeelColor()
    bin_num, _ = a.calculate_bin_distance([-20.0, 70.0], method="Y-component")
    assert 1 <= bin_num <= len(a.SCORE)
