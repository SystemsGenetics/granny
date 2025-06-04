from Granny.Models.AIModel.YoloModel import YoloModel

import pytest

# Where can I find the model so further testing can happen
# def test_load_getModel():
#     model = YoloModel()
#     model.loadModel

#Confirms the constructor correctly stores the model directory.
def test_yolomodel_init():
    model_path = "models/fake_yolo.pt"
    model = YoloModel(model_path)
    assert model.model_dir == model_path


def test_yolomodel_load_and_get_model(monkeypatch):
    # Mock the YOLO class to avoid loading a real model file
    class MockYOLO:
        def __init__(self, model_dir):
            self.path = model_dir

    # Replace YOLO in the YoloModel module with MockYOLO
    monkeypatch.setattr("Granny.Models.AIModel.YoloModel.YOLO", MockYOLO)

    model_path = "models/fake_yolo.pt"
    model = YoloModel(model_path)
    model.loadModel()

    assert isinstance(model.getModel(), MockYOLO)
    assert model.getModel().path == model_path