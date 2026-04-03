import pytest
from Granny.Analyses.Segmentation import Segmentation


def test_performAnalysis():
    analysis = Segmentation()


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

def test_default_conf_threshold_is_0_7():
    assert Segmentation().conf_threshold.getValue() == pytest.approx(0.7)


def test_default_iou_threshold():
    assert Segmentation().iou_threshold.getValue() == pytest.approx(0.45)


def test_default_font_scale():
    s = Segmentation()
    assert s.font_scale.getValue() == pytest.approx(2.0)


def test_default_text_thickness_is_3():
    assert Segmentation().text_thickness.getValue() == 3


def test_default_text_color_is_black():
    s = Segmentation()
    assert s.text_color_r.getValue() == 0
    assert s.text_color_g.getValue() == 0
    assert s.text_color_b.getValue() == 0


def test_text_color_boundary_values():
    s = Segmentation()
    s.text_color_r.setValue(255)
    s.text_color_g.setValue(255)
    s.text_color_b.setValue(255)
    assert s.text_color_r.getValue() == 255
    assert s.text_color_g.getValue() == 255
    assert s.text_color_b.getValue() == 255


def test_conf_threshold_range():
    s = Segmentation()
    s.conf_threshold.setValue(0.0)
    assert s.conf_threshold.getValue() == pytest.approx(0.0)
    s.conf_threshold.setValue(1.0)
    assert s.conf_threshold.getValue() == pytest.approx(1.0)


def test_qr_detector_initialized():
    from Granny.Utils.QRCodeDetector import QRCodeDetector
    s = Segmentation()
    assert isinstance(s.qr_detector, QRCodeDetector)


def test_variety_info_initially_none():
    assert Segmentation().variety_info is None


def test_analysis_name_is_segmentation():
    assert Segmentation().__analysis_name__ == "segmentation"
