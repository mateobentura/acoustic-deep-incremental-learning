import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import segmentation_models as sm


def to_one_hot(image, label):
    global classes
    label = tf.one_hot(label, classes, name='label', axis=-1)
    return image, label


def crops_to_dataset(crops, labels, balanced=False, split=False, shuffle=True):
    images = crops.reshape((crops.shape[0]*crops.shape[1], crops.shape[2], crops.shape[3]))
    lbs = labels.reshape(-1).astype(int)
    ds = tf.data.Dataset.from_tensor_slices((images, lbs))
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(ds_size)

    if balanced:
        # Balancing
        bg = np.array([bool(data[1] == 0) for data in ds])
        count = np.bincount(bg)
        ds = ds.batch(32)
        ds_bg = (
                    ds
                    .unbatch()
                    .filter(lambda features, label: label == 0)
                    .repeat())
        ds_obj = (
                    ds
                    .unbatch()
                    .filter(lambda features, label: label == 1)
                    .repeat())

        ds = tf.data.experimental.sample_from_datasets(
            [ds_bg.take(count[0]), ds_obj], [0.5, 0.5])

        ds_size = len(ds.take(count[0]*2))

    # ds = ds.map(to_one_hot)
    if split:
        split = {'train': 0.8, 'val': 0.2}

        for i in split:
            split[i] = int(split[i] * ds_size)

        ds_train = ds.take(split['train'])
        ds_val = ds.take(split['val'])

        ds_train = ds_train.prefetch(32)
        ds_val = ds_val.prefetch(8)
        return ds_train, ds_val
    else:
        return ds.prefetch(32)


class ClassifModel(keras.Model):
    """Define custom Model class to edit training stage."""
    def train_step(self, image, window_size=32, pad_h=1, pad_v=1):
        """Define custom training step."""
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, _, _ = image.sliding_window(window_size, pad_h, pad_v, 0.9)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def classification_model(img_shape, fine_tune_layers=0, dropout=False):
    base_model = keras.applications.ResNet50(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(32, 32, 3),
      include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    if fine_tune_layers >= 1:
        # Freeze all the layers except for the last `fine_tune_layers`
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
    else:
        base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=img_shape+(1,))
    x = inputs
    if img_shape[0] != 32:
        x = keras.layers.experimental.preprocessing.Resizing(32,32)(x)
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
    outputs = keras.layers.Dense(1)(x)
    model = ClassifModel(inputs, outputs)

    opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    loss = keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model


def segmentation_model(img_shape, backbone='resnet34', classes=1):
    inputs = keras.Input(shape=img_shape+(1,))
    x = keras.layers.Conv2D(3,(3,3), padding='same')(inputs)
    base_model = sm.Unet(backbone_name=backbone, classes=classes, input_shape=img_shape+(3,), encoder_weights='imagenet', encoder_freeze=True)
    outputs = base_model(x)
    model = keras.Model(inputs, outputs, name=base_model.name)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
