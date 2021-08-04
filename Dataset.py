import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import segmentation_models as sm
from skimage.util import view_as_windows
import matplotlib.pyplot as plt


def specificity(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels
    Returns:
    Specificity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

METRICS = [
    dice_coef,
    keras.metrics.BinaryAccuracy(name='acc'),
    keras.metrics.Recall(name='sensitivity'),
    specificity
]


def classification_model(img_shape, classes=1, dropout=False):
    base_model = keras.applications.ResNet50(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(32, 32, 3),
      include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    base_model.trainable = False

    input = keras.Input(shape=img_shape+(1,))
    x = input
    if img_shape[0] != 32:
        x = keras.layers.experimental.preprocessing.Resizing(32, 32)(x)
    # Convolve to adapt to 3-channel input
    x = keras.layers.Conv2D(3, (3, 3), padding='same')(x)
    # Pre-processing
    x = keras.applications.resnet50.preprocess_input(x)
    # Base pre-trained model
    x = base_model(x, training=False)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(256, activation='relu')(x)
    if dropout: x = keras.layers.Dropout(0.7)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    if dropout: x = keras.layers.Dropout(0.7)(x)
    output = keras.layers.Dense(classes+1, name='Classification', activation='softmax')(x)
    model = keras.Model(input, output)

    opt = keras.optimizers.Adam(learning_rate=1e-5)
    #loss = keras.losses.CategoricalCrossentropy()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=METRICS)
    return model


def segmentation_model(img_shape, classes=1, backbone='resnet34'):
    input = keras.Input(shape=img_shape+(1,))
    x = keras.layers.Conv2D(3, (3, 3), padding='same')(input)
    base_model = sm.Unet(backbone_name=backbone,
                        classes=classes,
                        input_shape=img_shape+(3,),
                        activation='sigmoid',
                        encoder_weights='imagenet',
                        encoder_freeze=False)
    output = base_model(x)
    base_model._name = 'Segmentation'
    model = keras.Model(input, output, name=base_model.name)
    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=opt, loss=sm.losses.bce_dice_loss, metrics=METRICS)
    return model
