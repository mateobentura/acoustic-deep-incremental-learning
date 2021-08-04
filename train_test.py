import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import ImageSynthetique as imsy
from ImageSynthetique import timing
from tensorflow import keras
import Dataset as ds
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
img_dir = 'images/'

def main():
    """Run main."""
    seed = 30
    test = imsy.Image(640, width=640, noise_lvl=1e-3, seed=seed)
    spacings = [5, 7, 9, 11, 13, 15]
    for spacing in spacings:
        pt = [47*(spacing-3), 60*(spacing-4)]
        test.add_ladder(starting_pt=pt,
                        spacing=spacing, length=12,
                        l_var=2, lines=4*(55//spacing), seed=seed)
    test.finish()
    test.plot_label()
    plt.savefig('test')
    ds_train, ds_val = test.crops_to_dataset(split=True)
    

if __name__ == "__main__":
    main()
