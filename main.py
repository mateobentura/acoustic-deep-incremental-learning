import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import ImageSynthetique as imsy
from ImageSynthetique import timing
from tensorflow import keras
import Dataset as ds
import sys
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """Run main."""
    height, width = (300, 640)
    threshold = float(sys.argv[2])
    window_size = 32
    img_dir = 'images/'
    BACKBONE = 'resnet34'
    var_time = timing()
    if sys.argv[1] == 'train': train(height, width, img_dir, threshold)
    elif sys.argv[1] == 'test_gen':
        test = imsy.Image(300, noise_lvl=threshold)
        # Premier objet
        spacings = [5, 7, 9, 11, 13, 15]
        for spacing in spacings:
            pt = [47*(spacing-3), 30]
            test.add_ladder(starting_pt=pt,
                            spacing=spacing, length=12,
                            l_var=2, lines=4*(55//spacing))
        test.finish()
        test.plot_label()
        plt.savefig('test_gen')
        plt.figure()
        plt.hist(test.image.reshape(-1), bins=20)
        plt.savefig('test_hist')
    elif sys.argv[1] == 'test':
        # Image test
        seed = 500
        img_dir = 'noise_test/'
        # noise_min, noise_max = float(sys.argv[3]), float(sys.argv[4])
        # range = np.arange(noise_min, noise_max+0.05, 0.05).round(2)
        range = [0.05, 0.10, 0.12, 0.25, 0.35]
        img_shape = (window_size, window_size)
        var_time = timing()
        # Define models
        # Classification
        classif_model = ds.classification_model(img_shape)
        classif_model.load_weights('weights/classif_noise/classif')
        # Segmenation
        segm_model = ds.segmentation_model(img_shape, backbone=BACKBONE)
        segm_model.load_weights('weights/segm_noise/segm')
        m_classif = {'sensibilité': np.zeros((len(range))), 'specificité': np.zeros(len(range))}
        m_segm = {'sensibilité': np.zeros(len(range)), 'specificité': np.zeros(len(range))}
        var_time = timing('load models', var_time)
        index = 0
        for noise_lvl in range:
            print('================================================================')
            test = imsy.Image(300, noise_lvl=noise_lvl, seed=seed)
            # Premier objet
            spacings = [5, 7, 9, 11, 13, 15]
            for spacing in spacings:
                pt = [47*(spacing-3), 30]
                test.add_ladder(starting_pt=pt,
                                spacing=spacing, length=12,
                                l_var=2, lines=4*(55//spacing), seed=seed)
            # print(test.image[test.image>255].size)
            test.finish()
            test.plot_label()
            niv_str = ('%.2f' % noise_lvl).replace('.', '_')
            plt.savefig(img_dir+'test_'+niv_str)
            plt.close()
            var_time = timing('test image generation', var_time)

            pad_h = 16
            pad_v = 16
            test.sliding_window(window_size, pad_h, pad_v, threshold)
            ds_test = ds.crops_to_dataset(test.crops, test.labels['classif'][:, :, 0], shuffle=False)
            test.calssification_predict(classif_model, ds_test, test.labels['classif'][:, :, 0].shape, threshold)
            plt.savefig(img_dir+'test_classif_'+niv_str)
            plt.close()
            var_time = timing('classification prediction for noise level '+str(noise_lvl), var_time)
            print('================================')
            _, m_classif['sensibilité'][index], m_classif['specificité'][index] = test.confusion_matrix('classif')

            var_time = timing()
            # SEGMENTATION
            test.sliding_window(window_size, pad_h=window_size, pad_v=window_size, threshold=0.8)
            # Predict
            test.segmentation_predict(segm_model, test.crops, threshold=0.99)
            plt.savefig(img_dir+'test_segm_'+niv_str)
            _, m_segm['sensibilité'][index], m_segm['specificité'][index] = test.confusion_matrix('segm')
            var_time = timing('segmentation prediction for noise level '+str(noise_lvl), var_time)

            index += 1

            var_time = timing('test_segm', var_time)
        np.savetxt('m_classif', np.array([m_classif['sensibilité'], m_classif['specificité']]))
        np.savetxt('m_segm', np.array([m_segm['sensibilité'], m_segm['specificité']]))
    elif sys.argv[1] == 'test_sens':
        m = np.loadtxt('m_classif')
        m_classif = {}
        m_classif['sensibilité'], m_classif['specificité'] = m[0], m[1]
        m = np.loadtxt('m_segm')
        m_segm = {}
        m_segm['sensibilité'], m_segm['specificité'] = m[0], m[1]
        noise_min, noise_max = float(sys.argv[2]), float(sys.argv[3])
        range = np.arange(noise_min, noise_max+0.05, 0.05).round(2)
        # plot_metrics(range, m_classif, m_segm)
        plt.savefig(img_dir+'sens_spec')


def train(height, width, img_dir, threshold):
    # Image test
    var_time = timing()
    train = imsy.Image(height, width)
    train.add_ladder(starting_pt=[24 * 4, 25],
                    spacing=7, length=12, l_var=1, lines=32)
    train.add_ladder(starting_pt=[350, 30],
                    spacing=13, length=12,
                    l_var=1, lines=18)
    train.plot_label()
    plt.savefig(img_dir + 'train')
    var_time = timing('train', var_time)

    # Sliding window
    window_size = 32
    pad_h = 1
    pad_v = 1
    img_shape = (window_size, window_size)
    # Sliding window
    train.sliding_window(window_size, pad_h, pad_v, threshold)
    var_time = timing('sliding_window', var_time)

    ds_train, ds_val = ds.crops_to_dataset(crops, labels, balanced=False, split=True)
    var_time = timing('crops_to_dataset', var_time)

    classif_model = ds.classification_model(img_shape)

    epochs = 5
    classif_model.fit(ds_train.batch(64),
                      validation_data=ds_val.batch(8),
                      epochs=epochs,
                      verbose=1)
    var_time = timing('classification model training', var_time)
    classif_model.save_weights('classif/classif')


if __name__ == "__main__":
    main()
