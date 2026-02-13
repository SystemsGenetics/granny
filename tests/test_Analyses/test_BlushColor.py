from Granny.Analyses.BlushColor import BlushColor


def test_BlushColorInstantiation():
    """Test that BlushColor can be instantiated"""
    analysis = BlushColor()
    assert analysis is not None
    assert analysis.__analysis_name__ == "blush"


def test_BlushColorInputImages():
    """Test that BlushColor input_images can be set"""
    analysis = BlushColor()
    analysis.input_images.setValue("demo/pear_images/full_masked_images")
    assert analysis.input_images.getValue() == "demo/pear_images/full_masked_images"
