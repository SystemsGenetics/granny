import numpy as np
import pytest
from Granny.Analyses.SuperficialScald import SuperficialScald


def test_SuperficialScaldInstantiation():
    analysis = SuperficialScald()
    assert analysis is not None
    assert analysis.__analysis_name__ == "scald"


def test_SuperficialScaldInputImages():
    analysis = SuperficialScald()
    analysis.input_images.setValue("demo/granny_smith_images/full_masked_images")
    assert analysis.input_images.getValue() == "demo/granny_smith_images/full_masked_images"


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

def test_default_morph_kernel_is_10():
    assert SuperficialScald().morph_kernel.getValue() == 10


def test_default_min_threshold_is_100():
    assert SuperficialScald().min_threshold.getValue() == 100


def test_default_purple_threshold_is_126():
    assert SuperficialScald().purple_threshold.getValue() == 126


def test_default_blur_kernel_is_3():
    assert SuperficialScald().blur_kernel.getValue() == 3


def test_default_hist_factor():
    assert SuperficialScald().hist_factor.getValue() == pytest.approx(0.333)


def test_default_hist_top_n_is_10():
    assert SuperficialScald().hist_top_n.getValue() == 10


# ---------------------------------------------------------------------------
# _smoothMask
# ---------------------------------------------------------------------------

def test_smooth_mask_output_shape():
    a = SuperficialScald()
    mask = np.ones((100, 100), dtype=np.uint8)
    result = a._smoothMask(mask)
    assert result.shape == mask.shape


def test_smooth_mask_all_zeros_stays_zero():
    a = SuperficialScald()
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = a._smoothMask(mask)
    assert result.sum() == 0


def test_smooth_mask_all_ones_stays_ones():
    a = SuperficialScald()
    mask = np.ones((100, 100), dtype=np.uint8)
    result = a._smoothMask(mask)
    assert result.sum() > 0


# ---------------------------------------------------------------------------
# _calculateScald
# ---------------------------------------------------------------------------

def test_calculate_scald_full_mask_returns_zero():
    a = SuperficialScald()
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    bw = img.copy()
    score = a._calculateScald(bw, img)
    assert score == pytest.approx(0.0)


def test_calculate_scald_empty_mask_returns_one():
    a = SuperficialScald()
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    bw = np.zeros((50, 50, 3), dtype=np.uint8)
    score = a._calculateScald(bw, img)
    assert score == pytest.approx(1.0)


def test_calculate_scald_returns_value_between_0_and_1():
    a = SuperficialScald()
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    bw = img.copy()
    bw[:25, :] = 0
    score = a._calculateScald(bw, img)
    assert 0.0 <= score <= 1.0


def test_calculate_scald_zero_ground_area_returns_one():
    a = SuperficialScald()
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    bw = np.zeros((50, 50, 3), dtype=np.uint8)
    score = a._calculateScald(bw, img)
    assert score == 1


# ---------------------------------------------------------------------------
# _removeTrayResidue
# ---------------------------------------------------------------------------

def test_remove_tray_residue_output_shape():
    a = SuperficialScald()
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    result = a._removeTrayResidue(img)
    assert result.shape == img.shape


def test_remove_tray_residue_does_not_modify_input():
    a = SuperficialScald()
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    original = img.copy()
    a._removeTrayResidue(img)
    np.testing.assert_array_equal(img, original)
