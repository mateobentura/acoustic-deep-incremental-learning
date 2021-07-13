import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import ImageSynthetique as imsy
from ImageSynthetique import timing
from tensorflow import keras
import keras_visualizer as kv
import Dataset as ds
import sys
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """Run main."""
    height, width = (300, 640)
    train = imsy.Image(height, width)
    train.add_ladder(starting_pt=[24 * 4, 25],
                    spacing=7, length=12, l_var=1, lines=32)
    img_shape = (32, 32)
    BACKBONE = 'resnet34'
    meta_model = ds.meta_model(img_shape)
    # meta_model.save('meta_model.h5')

if __name__ == "__main__":
    main()
