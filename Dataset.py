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


METRICS = [
      keras.metrics.BinaryAccuracy(name='acc'),
      keras.metrics.Recall(name='sensitivity'),
      specificity
]


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


def crops_to_dataset(crops, labels, classes=1, balanced=False, split=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((
                crops.reshape(-1,32,32,1),
                keras.utils.to_categorical(labels.reshape(-1), classes+1)
                ))
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

    def to_one_hot(image, label):
        nonlocal classes
        label = tf.one_hot(label, classes+1, name='label', axis=-1)
        return image, label

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


class MetaModel(keras.Model):
    """Define custom Model class to edit training stage."""
    # def train_step(self, image, window_size=32, pad_h=1, pad_v=1):
    #     """Define custom training step."""
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #
    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     # Update metrics (includes the metric that tracks the loss)
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}

    def test_train(self, data, epochs=1):
        x, y = data
        test_step(data)

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * 64))

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

def meta_model(img_shape, classes=1, backbone='resnet34'):
    # INPUT
    input = keras.Input(shape=img_shape+(1,))

    # CLASSIFICATION MODEL
    base_model = keras.applications.ResNet50(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(32, 32, 3),
      include_top=False,
    )  # Do not include the ImageNet classifier at the top.
    base_model.trainable = False

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
    x = keras.layers.Dense(256, activation='relu')(x)
    out_classif = keras.layers.Dense(classes+1, name='Classification', activation='softmax')(x)

    # SEGMENTATION MODEL
    x = keras.layers.Conv2D(3, (3, 3), padding='same')(input)
    base_model = sm.Unet(backbone_name=backbone, classes=classes, input_shape=img_shape+(3,), encoder_weights='imagenet', encoder_freeze=False)
    out_segm = base_model(x)
    base_model._name = 'Segmentation'

    model = keras.Model(input, [out_classif, out_segm], name='meta_model')
    model.compile(
    optimizer='adam',
    loss={
        'Classification': keras.losses.CategoricalCrossentropy(),
        'Segmentation': keras.losses.BinaryCrossentropy(),
    },
    metrics = ['accuracy']
    )

    return model


def classification_model(img_shape, classes=1, fine_tune_layers=0, dropout=False):
    base_model = keras.applications.ResNet50(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(32, 32, 3),
      include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    if fine_tune_layers > 0:
        # Freeze all the layers except for the last `fine_tune_layers`
        for layer in base_model.layers[:-fine_tune_layers]:
          layer.trainable =  False
    else:
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
    model.compile(optimizer=opt, loss=sm.losses.bce_dice_loss, metrics=[dice_coef]+METRICS)
    return model

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image, batch_size=32, window_size=32, step=1, network='classif', test=False):
        self.network = network
        self.image = image
        assert batch_size <= image.width-window_size, 'Batch size must be less or equal to width minus window size.'
        self.batch_size = batch_size
        self.window_size = window_size
        self.step = step
        self.mask = np.zeros_like(self.image.image)

    def __len__(self):
        return np.ceil(((self.image.height-self.window_size)*(self.image.width-self.window_size))/(self.batch_size*self.step**2)).astype(int)

    def __getitem__(self, idx):
        # print(idx)
        y, x = np.unravel_index([(idx*self.batch_size*self.step**2), ((idx+1)*self.batch_size*self.step**2)-1], (self.image.height-self.window_size, self.image.width-self.window_size))
        # print('\n')
        # print(((x[0],y[0]),(x[1], y[1])))
        if x[0] > x[1]:
            x[1] = self.image.width-self.window_size - 1

        batch_subimage = self.image.image[y[0]:y[0]+self.window_size, x[0]:x[1]+self.window_size+1].copy()
        # print(batch_subimage.shape)
        # if x[0] > 500:
        #     print('\n')
        #     print(((x[0],y[0]),(x[1], y[1])))
        # plt.imshow(batch_subimage)
        # plt.show()
        self.mask[y[0]:y[0]+1, x[0]:x[1]+1] +=1
        batch_x = view_as_windows(batch_subimage, self.window_size, step=self.step)
        batch_x = np.reshape(batch_x, (batch_x.shape[1], self.window_size, self.window_size, 1))
        rng = np.random.RandomState()
        indexes = np.arange(0, batch_x.shape[0]-1)
        rng.shuffle(indexes)
        batch_x = batch_x[indexes]
        batch_x /= 255.
        if self.network == 'classif':
            batch_c = self.image.mask[y[0]:y[0]+self.window_size, x[0]:x[1]+self.window_size].copy()
            batch_c = view_as_windows(batch_c, self.window_size, step=self.step).squeeze()
            batch_c = (batch_c>0).any(axis=(-2,-1)).astype(int)
            batch_c = batch_c[indexes]
            # print(batch_c.shape)
            # batch_y = tf.one_hot(batch_c, self.image.classes+1)
            batch_y = keras.utils.to_categorical(batch_c, self.image.classes+1)
        elif self.network == 'segm':
            preprocessing_fn = sm.get_preprocessing('resnet34')
            batch_x = preprocessing_fn(batch_x)
            batch_s = self.image.segmentation[y[0]:y[0]+self.window_size, x[0]:x[1]+self.window_size].copy()
            batch_y = view_as_windows(batch_s, (self.window_size, self.window_size, self.image.classes), step=(self.step, self.step, 1))
            # if batch_y.shape[1] != self.batch_size:
            #     print('\n')
            #     print(batch_y.shape)
            batch_y = np.reshape(batch_y, (batch_y.shape[1], self.window_size, self.window_size, self.image.classes))
            batch_y = batch_y[indexes]
            batch_y /= 255.
        else:
            assert True, 'Error'
            pass
        return batch_x, batch_y
