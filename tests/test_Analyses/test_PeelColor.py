from Granny.Analyses.PeelColor import PeelColor


def test_PeelColorInstantiation():
    """Test that PeelColor can be instantiated"""
    analysis = PeelColor()
    assert analysis is not None
    assert analysis.__analysis_name__ == "color"


def test_PeelColorInputImages():
    """Test that PeelColor input_images can be set"""
    analysis = PeelColor()
    analysis.input_images.setValue("demo/pear_images/full_masked_images")
    assert analysis.input_images.getValue() == "demo/pear_images/full_masked_images"
