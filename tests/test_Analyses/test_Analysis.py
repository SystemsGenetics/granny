from Granny.Analyses.StarchArea import StarchArea
from Granny.Models.Images.RGBImage import RGBImage


def _get_analysis():
    """Use StarchArea as a concrete implementation of Analysis."""
    return StarchArea()


def test_parse_qr_from_filename_valid():
    analysis = _get_analysis()
    result = analysis._parse_qr_from_filename(
        "APPLE2025_LOT001_2025-12-02_BB-Late_fruit_01.png"
    )
    assert result["project"] == "APPLE2025"
    assert result["lot"] == "LOT001"
    assert result["date"] == "2025-12-02"
    assert result["variety"] == "BB-Late"


def test_parse_qr_from_filename_with_path():
    analysis = _get_analysis()
    result = analysis._parse_qr_from_filename(
        "/some/path/to/APPLE2025_LOT001_2025-12-02_BB-Late_fruit_05.png"
    )
    assert result["project"] == "APPLE2025"
    assert result["variety"] == "BB-Late"


def test_parse_qr_from_filename_jpg():
    analysis = _get_analysis()
    result = analysis._parse_qr_from_filename(
        "PROJ_LOT_DATE_VAR_fruit_01.jpg"
    )
    assert result["project"] == "PROJ"
    assert result["variety"] == "VAR"


def test_parse_qr_from_filename_legacy():
    """Legacy filenames without QR data should return empty strings."""
    analysis = _get_analysis()
    result = analysis._parse_qr_from_filename("apple_fruit_01.png")
    assert result["project"] == ""
    assert result["lot"] == ""
    assert result["date"] == ""
    assert result["variety"] == ""


def test_parse_qr_from_filename_no_match():
    analysis = _get_analysis()
    result = analysis._parse_qr_from_filename("random_image.png")
    assert result["project"] == ""


def test_add_qr_metadata_valid():
    analysis = _get_analysis()
    img = RGBImage("APPLE2025_LOT001_2025-12-02_BB-Late_fruit_01.png")
    analysis._add_qr_metadata(img, "APPLE2025_LOT001_2025-12-02_BB-Late_fruit_01.png")

    metadata = img.getMetaData()
    assert "project" in metadata
    assert metadata["project"].getValue() == "APPLE2025"
    assert metadata["lot"].getValue() == "LOT001"
    assert metadata["date"].getValue() == "2025-12-02"
    assert metadata["variety"].getValue() == "BB-Late"


def test_add_qr_metadata_legacy():
    """Legacy filenames should not add QR metadata."""
    analysis = _get_analysis()
    img = RGBImage("apple_fruit_01.png")
    analysis._add_qr_metadata(img, "apple_fruit_01.png")

    metadata = img.getMetaData()
    assert "project" not in metadata
    assert "lot" not in metadata
