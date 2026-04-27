import os
import tempfile
import pandas as pd
from Granny.Models.Values.MetaDataValue import MetaDataValue
from Granny.Models.Values.FloatValue import FloatValue
from Granny.Models.Values.StringValue import StringValue
from Granny.Models.Images.RGBImage import RGBImage
import numpy as np


def _make_image(name, rating_val, project=None, lot=None, date=None, variety=None):
    """Helper to create a test image with metadata."""
    img = RGBImage(name)
    img.setImage(np.zeros((10, 10, 3), dtype=np.uint8))

    rating = FloatValue("rating", "rating", "test rating")
    rating.setMin(0.0)
    rating.setMax(1.0)
    rating.setValue(rating_val)
    img.addValue(rating)

    if project:
        for key, val in [("project", project), ("lot", lot), ("date", date), ("variety", variety)]:
            sv = StringValue(key, key, f"test {key}")
            sv.setValue(val)
            img.addValue(sv)

    return img


def test_write_tray_summary_with_string_columns():
    """tray_summary.csv should include string metadata columns like project, lot, date, variety."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mdv = MetaDataValue("results", "results", "test")
        mdv.setValue(tmpdir)

        images = [
            _make_image("PROJ_LOT1_2025-01-01_VAR_fruit_01.png", 0.8, "PROJ", "LOT1", "2025-01-01", "VAR"),
            _make_image("PROJ_LOT1_2025-01-01_VAR_fruit_02.png", 0.6, "PROJ", "LOT1", "2025-01-01", "VAR"),
        ]
        mdv.setImageList(images)
        mdv.writeValue()

        tray_df = pd.read_csv(os.path.join(tmpdir, "tray_summary.csv"))
        assert "project" in tray_df.columns
        assert "lot" in tray_df.columns
        assert "date" in tray_df.columns
        assert "variety" in tray_df.columns
        assert tray_df["project"].iloc[0] == "PROJ"
        assert tray_df["lot"].iloc[0] == "LOT1"
        assert tray_df["rating"].iloc[0] == 0.7  # average of 0.8 and 0.6


def test_write_tray_summary_without_string_columns():
    """tray_summary.csv should still work without string metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mdv = MetaDataValue("results", "results", "test")
        mdv.setValue(tmpdir)

        images = [
            _make_image("apple_fruit_01.png", 0.9),
            _make_image("apple_fruit_02.png", 0.7),
        ]
        mdv.setImageList(images)
        mdv.writeValue()

        tray_df = pd.read_csv(os.path.join(tmpdir, "tray_summary.csv"))
        assert "TrayName" in tray_df.columns
        assert "rating" in tray_df.columns
        assert abs(tray_df["rating"].iloc[0] - 0.8) < 0.001


def test_results_csv_has_all_rows():
    """results.csv should have one row per image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mdv = MetaDataValue("results", "results", "test")
        mdv.setValue(tmpdir)

        images = [
            _make_image("PROJ_LOT1_2025-01-01_VAR_fruit_01.png", 0.5, "PROJ", "LOT1", "2025-01-01", "VAR"),
            _make_image("PROJ_LOT1_2025-01-01_VAR_fruit_02.png", 0.6, "PROJ", "LOT1", "2025-01-01", "VAR"),
            _make_image("PROJ_LOT1_2025-01-01_VAR_fruit_03.png", 0.7, "PROJ", "LOT1", "2025-01-01", "VAR"),
        ]
        mdv.setImageList(images)
        mdv.writeValue()

        results_df = pd.read_csv(os.path.join(tmpdir, "results.csv"))
        assert len(results_df) == 3
