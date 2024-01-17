from matplotlib import pyplot as plt
from models import *
import keras
from harness import run_test_harness

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs_mlp/", histogram_freq=1, write_images=True)
run_test_harness(MLP(), tensorboard_callback, augmentation=False)
print("Done")