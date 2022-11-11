import datetime as datetime 
from surprise import dump
class utils():
    def serialize_model(self, model):
        model_name = "model_" + str(datetime.datetime.now()) + ".surp"
        dump.dump(model_name, predictions=None, algo=model)
        #sanjana_mlip_project/models
        return model_name
