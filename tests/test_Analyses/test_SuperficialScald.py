from Granny.Analyses.SuperficialScald import SuperficialScald


def test_SuperficialScaldInstantiation():
    """Test that SuperficialScald can be instantiated"""
    analysis = SuperficialScald()
    assert analysis is not None
    assert analysis.__analysis_name__ == "scald"


def test_SuperficialScaldInputImages():
    """Test that SuperficialScald input_images can be set"""
    analysis = SuperficialScald()
    analysis.input_images.setValue("demo/granny_smith_images/full_masked_images")
    assert analysis.input_images.getValue() == "demo/granny_smith_images/full_masked_images"
