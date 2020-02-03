from keras.models import load_model

from training.train import Train
from utils.utils import Utils

model = load_model("model.h5")
TEXT = "is there ai dept?"
train = Train()
train.predictions()
