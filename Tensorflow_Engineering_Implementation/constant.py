
import os

_LEARN_HOME = r"F:\for learn\Python\tensorflow_engineering_implementation"
SOURCE_HOME = os.path.join(_LEARN_HOME, "data")
PRE_TRAINED_MODEL_HOME = r"F:\for learn\Python\pre-trained-models"
OUTPUT_HOME = os.path.join(_LEARN_HOME, "output")


for path in [OUTPUT_HOME, PRE_TRAINED_MODEL_HOME]:
    os.makedirs(path, exist_ok=True)

