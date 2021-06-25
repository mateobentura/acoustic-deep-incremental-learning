import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import segmentation_models as sm

def to_one_hot(image, label):
    global classes
    label = tf.one_hot(label, classes, name='label', axis=-1)
    return image, label

def crops_to_dataset(crops, labels, balanced=True, split=False, shuffle=True):
    images = crops.reshape((crops.shape[0]*crops.shape[1], crops.shape[2], crops.shape[3]))
    lbs = labels.reshape(-1).astype(int)
    ds = tf.data.Dataset.from_tensor_slices((images,lbs))
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
                    .filter(lambda features, label: label==0)
                    .repeat())
        ds_obj = (
                    ds
                    .unbatch()
                    .filter(lambda features, label: label==1)
                    .repeat())

        ds = tf.data.experimental.sample_from_datasets(
            [ds_bg.take(count[0]), ds_obj], [0.5, 0.5])

        ds_size = len(ds.take(count[0]*2))

    #ds = ds.map(to_one_hot)
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


def classification_model(img_shape, fine_tune_layers=0, dropout=False):
    base_model = keras.applications.ResNet50(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(32,32,3),
      include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    if fine_tune_layers >= 1:
        # Freeze all the layers except for the last `fine_tune_layers`
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable =  False
    else:
        base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=img_shape)
    x = inputs
    if img_shape[0] != 32:
        x = keras.layers.experimental.preprocessing.Resizing(32,32)(x)
    # Convolve to adapt to 3-channel input
    x = keras.layers.Conv2D(3,(3,3), padding='same')(x)
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
    model = keras.Model(inputs, outputs)

    opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    loss = keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model

def segmentation_model(img_shape):
    model = sm.Unet(backbone_name='resnet34', classes=1, input_shape=img_shape, encoder_freeze=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
