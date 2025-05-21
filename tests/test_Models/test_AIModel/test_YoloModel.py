from Granny.Models.AIModel.YoloModel import YoloModel

# Where can I find the model so further testing can happen
# def test_load_getModel():
#     model = YoloModel()
#     model.loadModel

#Confirms the constructor correctly stores the model directory.
def test_yolomodel_init():
    model_path = "models/fake_yolo.pt"
    model = YoloModel(model_path)
    assert model.model_dir == model_path