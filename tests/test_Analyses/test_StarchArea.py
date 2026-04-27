import numpy as np
import pytest
from Granny.Analyses.StarchArea import StarchArea, StarchScales, load_starch_scales


# ---------------------------------------------------------------------------
# load_starch_scales / StarchScales
# ---------------------------------------------------------------------------

def test_load_starch_scales_returns_dict():
    """YAML loads as a non-empty dictionary."""
    data = load_starch_scales()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_load_starch_scales_expected_varieties():
    """All expected varieties are present in the YAML."""
    data = load_starch_scales()
    expected = {
        "HONEYCRISP", "WA38_1", "WA38_2", "ENZA",
        "CORNELL", "PURDUE", "DANJOU",
        "GOLDEN_DELICIOUS", "GRANNY_SMITH", "JONAGOLD",
    }
    assert expected.issubset(data.keys())


def test_load_starch_scales_index_rating_same_length():
    """Every variety has equal-length index and rating lists."""
    data = load_starch_scales()
    for variety, scale in data.items():
        assert len(scale["index"]) == len(scale["rating"]), (
            f"{variety}: index length {len(scale['index'])} != "
            f"rating length {len(scale['rating'])}"
        )


def test_load_starch_scales_ratings_between_0_and_1():
    """All rating values are valid percentages (0.0-1.0)."""
    data = load_starch_scales()
    for variety, scale in data.items():
        for r in scale["rating"]:
            assert 0.0 <= r <= 1.0, f"{variety} has out-of-range rating: {r}"


def test_starch_scales_class_attributes():
    """StarchScales class exposes variety names as attributes."""
    data = load_starch_scales()
    for variety in data:
        assert hasattr(StarchScales, variety), f"StarchScales missing attribute: {variety}"


# ---------------------------------------------------------------------------
# StarchArea initialisation & parameters
# ---------------------------------------------------------------------------

def test_starch_area_instantiation():
    analysis = StarchArea()
    assert analysis is not None


def test_default_threshold_is_140():
    analysis = StarchArea()
    assert analysis.starch_threshold.getValue() == 140


def test_threshold_accepts_boundary_values():
    """Threshold should accept the full 0-255 range without error."""
    analysis = StarchArea()
    analysis.starch_threshold.setValue(0)
    assert analysis.starch_threshold.getValue() == 0
    analysis.starch_threshold.setValue(255)
    assert analysis.starch_threshold.getValue() == 255


def test_default_blur_kernel_is_7():
    analysis = StarchArea()
    assert analysis.blur_kernel.getValue() == 7


def test_default_mask_alpha_is_0_6():
    analysis = StarchArea()
    assert analysis.mask_alpha.getValue() == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# _calculateStarch - synthetic image tests
# ---------------------------------------------------------------------------

def _make_bgr(gray_value: int, size: int = 100) -> np.ndarray:
    """Create a solid-colour BGR image with a given grayscale brightness."""
    channel = np.full((size, size), gray_value, dtype=np.uint8)
    return np.stack([channel, channel, channel], axis=-1)


def test_calculate_starch_all_dark_returns_high_ratio():
    """A very dark image should be almost entirely starch."""
    analysis = StarchArea()
    img = _make_bgr(10)
    ratio, _ = analysis._calculateStarch(img)
    assert ratio > 0.8, f"Expected ratio > 0.8 for dark image, got {ratio}"


def test_calculate_starch_bright_dominant_image_returns_low_ratio():
    """An image that is mostly bright should detect low starch."""
    size = 100
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    # Small dark patch (10x10) in the corner - starch
    img[:10, :10] = 10

    analysis = StarchArea()
    ratio, _ = analysis._calculateStarch(img)
    assert ratio < 0.2, f"Expected ratio < 0.2 for mostly-bright image, got {ratio}"


def test_calculate_starch_returns_ratio_between_0_and_1():
    """Starch ratio must always be a valid proportion."""
    analysis = StarchArea()
    img = _make_bgr(128)
    ratio, _ = analysis._calculateStarch(img)
    assert 0.0 <= ratio <= 1.0


def test_calculate_starch_returns_image_same_shape():
    """The returned image must have the same shape as the input."""
    analysis = StarchArea()
    img = _make_bgr(100, size=64)
    _, result = analysis._calculateStarch(img)
    assert result.shape == img.shape


def test_calculate_starch_higher_threshold_gives_higher_ratio():
    """Raising the threshold should classify more pixels as starch."""
    img = _make_bgr(150)
    low_analysis = StarchArea()
    low_analysis.starch_threshold.setValue(80)   # ~31%
    high_analysis = StarchArea()
    high_analysis.starch_threshold.setValue(200)  # ~78%

    low_ratio, _ = low_analysis._calculateStarch(img)
    high_ratio, _ = high_analysis._calculateStarch(img)
    assert high_ratio >= low_ratio


def test_calculate_starch_mixed_image():
    """An image split between dark and bright halves should give ~50% ratio."""
    size = 100
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :50] = 10    # left half: very dark (starch)
    img[:, 50:] = 240   # right half: very bright (no starch)

    analysis = StarchArea()
    ratio, _ = analysis._calculateStarch(img)
    assert 0.3 < ratio < 0.7, f"Expected roughly 0.5, got {ratio}"


# ---------------------------------------------------------------------------
# _calculateIndex
# ---------------------------------------------------------------------------

def test_calculate_index_returns_all_varieties():
    """_calculateIndex should return a key for every loaded variety."""
    analysis = StarchArea()
    data = load_starch_scales()
    results = analysis._calculateIndex(0.5)
    assert set(results.keys()) == set(data.keys())


def test_calculate_index_exact_match():
    """When target exactly matches a rating, that index is selected."""
    analysis = StarchArea()
    # CORNELL index 5 -> rating 0.537191447
    target = 0.537191447
    results = analysis._calculateIndex(target)
    assert results["CORNELL"] == pytest.approx(5.0)


def test_calculate_index_closest_match():
    """A target slightly off a known rating should still pick the nearest index."""
    analysis = StarchArea()
    # HONEYCRISP index 1 -> rating 0.981640465
    results = analysis._calculateIndex(0.982)
    assert results["HONEYCRISP"] == pytest.approx(1.0)


def test_calculate_index_values_are_floats():
    """All returned index values should be numeric."""
    analysis = StarchArea()
    results = analysis._calculateIndex(0.5)
    for variety, index in results.items():
        assert isinstance(index, (int, float)), f"{variety} index is not numeric"


# ---------------------------------------------------------------------------
# _drawMask
# ---------------------------------------------------------------------------

def test_draw_mask_zeros_darken_pixels():
    """Pixels where mask == 0 should be darkened."""
    analysis = StarchArea()
    img = _make_bgr(200, size=10)
    mask = np.zeros((10, 10), dtype=np.uint8)  # all starch
    result = analysis._drawMask(img, mask)
    assert result.mean() < img.mean()


def test_draw_mask_ones_leave_pixels_unchanged():
    """Pixels where mask == 1 should be unchanged."""
    analysis = StarchArea()
    img = _make_bgr(200, size=10)
    mask = np.ones((10, 10), dtype=np.uint8)  # no starch
    result = analysis._drawMask(img, mask)
    np.testing.assert_array_equal(result, img)


def test_draw_mask_output_shape_matches_input():
    analysis = StarchArea()
    img = _make_bgr(128, size=50)
    mask = np.ones((50, 50), dtype=np.uint8)
    result = analysis._drawMask(img, mask)
    assert result.shape == img.shape