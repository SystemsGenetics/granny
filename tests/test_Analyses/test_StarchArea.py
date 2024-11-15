from Granny.Analyses.StarchArea import StarchArea 

def test_StarchAnalyses():
    analysis = StarchArea()
    analysis.input_images.value = "test-results/images/starch"
    analysis.performAnalysis()