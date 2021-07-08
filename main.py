import ImageSynthetique as imsy
import Dataset as ds
import sys
import time
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def timing(part='', start=None):
    """Time code sections.

    Arguments:
    ---------
        part (str): name of section
        start: previous timestamp

    """
    if start is not None:
        print("Part %s took %1.2fs" % (part, (time.time() - start)))
    return time.time()


def main():
    """Run main."""
    height, width = (300, 640)
    threshold = float(sys.argv[2])
    window_size = 32
    img_dir = 'images/'
    BACKBONE = 'resnet34'
    var_time = timing()
    if sys.argv[1] == 'train':
        # Image test
        train = imsy.Image(height, width)
        train.add_ladder(starting_pt=[24 * 4, 25],
                        spacing=7, length=12, l_var=1, lines=32)
        train.add_ladder(starting_pt=[350, 30],
                        spacing=13, length=12,
                        l_var=1, lines=18)
        train.plot_label()
        plt.savefig(img_dir + 'train')
        var_time = timing('train', var_time)

        # # Sliding window
        # window_size = 32
        # pad_h = 1
        # pad_v = 1

        # crops, labels, number, segmentation_crops = train.sliding_window(window_size, pad_h, pad_v, threshold)
        # var_time = timing('sliding_window', var_time)

        # indexes = (40//pad_v,50//pad_h-1)
        # plt.subplot(121)
        # plt.imshow(crops[indexes])
        # plt.subplot(122)
        # plt.imshow(segmentation_crops[indexes])
        # plt.savefig(img_dir+'crop')

        # train.compare_labels(labels, threshold)
        # var_time = timing('compare', var_time)
        # plt.savefig(img_dir+'compare')
        # var_time = timing('save', var_time)
        #
        # labels /= labels.max()
        # img_shape = (window_size,window_size,1)
        # ds_train, ds_val = crops_to_dataset(crops, labels, balanced=False, split=True)
    elif sys.argv[1] == 'test':
        # Image test
        seed = 500
        noise_min, noise_max = float(sys.argv[3]), float(sys.argv[4])
        range = np.arange(noise_min, noise_max+0.1, 0.1).round(1)
        img_shape = (window_size, window_size)
        # Define models
        # Classification
        classif_model = ds.classification_model(img_shape)
        classif_model.load_weights('weights/classif/classif')
        # Segmenation
        segm_model = ds.segmentation_model(img_shape, backbone=BACKBONE)
        segm_model.load_weights('weights/segm/segm')
        m_classif = {'sensibilité': np.zeros((6)), 'specificité': np.zeros((6))}
        m_segm = {'sensibilité': np.zeros((6)), 'specificité': np.zeros((6))}
        for noise_lvl in range:
            test = imsy.Image(300, noise_lvl=noise_lvl, seed=seed)
            # Premier objet
            spacings = [5, 7, 9, 11, 13, 15]
            for spacing in spacings:
                pt = [47*(spacing-3), 30]
                test.add_ladder(starting_pt=pt,
                                spacing=spacing, length=12,
                                l_var=2, lines=4*(55//spacing), seed=seed)
            test.plot_label()
            niv_str = str(noise_lvl).replace('.', '_')
            plt.savefig(img_dir+'test_'+niv_str)
            var_time = timing('test image generation', var_time)

            pad_h = 16
            pad_v = 16
            test.sliding_window(window_size, pad_h, pad_v, threshold)
            ds_test = ds.crops_to_dataset(test.crops, test.labels['classif'][:, :, 0], shuffle=False)
            test.calssification_predict(classif_model, ds_test, test.labels['classif'][:, :, 0].shape, threshold)
            plt.savefig(img_dir+'test_classif_'+niv_str)
            var_time = timing('classification prediction for noise level '+str(noise_lvl), var_time)

            index = int(noise_lvl*10) - 1
            _, m_classif['sensibilité'][index], m_classif['specificité'][index] = test.confusion_matrix('classif')

            var_time = timing()
            # Segmenation
            test.sliding_window(window_size, pad_h=window_size, pad_v=window_size, threshold=0.8)
            test.segmentation_predict(segm_model, test.crops, threshold=0.99)
            _, m_segm['sensibilité'][index], m_segm['specificité'][index] = test.confusion_matrix('segm')

            var_time = timing('test_segm', var_time)
            plt.savefig('test_predict_segm')
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
        range = np.arange(noise_min, noise_max+0.1, 0.1).round(1)

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8), constrained_layout=True, sharex=True)
        ax1.title.set_text('Classification')
        ax1.plot(range, m_classif['sensibilité'], label='Sensibilité')
        ax1.set_xlim(xmin=noise_min, xmax=noise_max)
        ax1.set_ylim(ymin=0.45, ymax=1)
        ax1.plot(range, m_classif['specificité'], label='Specificité')
        ax1.legend(loc='lower right')

        ax2.title.set_text('Segmentation')
        ax2.plot(range, m_segm['sensibilité'], label='Sensibilité')
        ax2.set_ylim(ymin=0.2, ymax=1)
        ax2.plot(range, m_segm['specificité'], label='Specificité')
        ax2.legend(loc='lower right')
        plt.savefig(img_dir+'sens_spec')

        plt.figure(figsize=(10, 5), constrained_layout=True)
        plt.gca().title.set_text('Classification')
        plt.plot(range, m_classif['sensibilité'], label='Sensibilité')
        plt.gca().set_xlim(xmin=noise_min, xmax=noise_max)
        plt.gca().set_ylim(ymin=0.8, ymax=1)
        plt.plot(range, m_classif['specificité'], label='Specificité')
        plt.legend(loc='lower right')



if __name__ == "__main__":
    main()
