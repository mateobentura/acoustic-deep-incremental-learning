import ImageSynthetique as imsy
from ImageSynthetique import timing
import Dataset as ds
import sys
import time
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """Run main."""
    height, width = (300, 640)
    train = imsy.Image(height, width)
    train.add_ladder(starting_pt=[24 * 4, 25],
                    spacing=7, length=12, l_var=1, lines=32)
    
