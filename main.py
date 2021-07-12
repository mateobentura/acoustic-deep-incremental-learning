import ImageSynthetique as imsy
import Dataset as ds
import sys
import time
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def timing(part='', start=None):
    """Time code sections.

    args:
        part - str
        start - timestamp
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
    var_time = timing()
    if sys.argv[1] == 'train':
        # Image test
        train = imsy.Image(height, width)
        train.add_ladder(starting_pt=[24 * 4, 25],
                        spacing=7, length=12, l_var=1, lines=32)
        train.add_ladder(starting_pt=[350, 30],
                        spacing=13, length=12,
                        l_var=1, lines=18)
        train.add_disk(center=[200, 200], radius=3, intensity=100)
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
    if sys.argv[1] == 'test':
        # Image test
        test = imsy.Image(300, noise_lvl=0.2)
        # Premier objet
        # spacings = [5, 7, 9, 11, 13, 15]
        # for spacing in spacings:
        #     pt = [47*(spacing-3), 30]
        #     test.add_ladder(starting_pt=pt,
        #                     spacing=spacing, length=12,
        #                     l_var=2, lines=4*(55//spacing), seed=40)
        # Add ladder
        test.add_ladder(starting_pt=[50, 30], spacing=7, length=12,
                        l_var=5, lines=30, seed=40)
        # Add disk
        test.add_disk(center=[180, 50], diameter=5, intensity=0.7)
        # Add lines
        test.add_lines(starting_pt=[320, 50], spacing=50, spacing_var=0.5, thickness=3,
                        lines=5, seed=30)
        test.add_vline(starting_pt=[400, 10], length=280, intensity=120)
        test.plot_label()
        plt.savefig(img_dir+'test')
        var_time = timing('test', var_time)
        # img_shape = (window_size, window_size, 1)
        # pad_h = 16
        # pad_v = 16
        # crops, labels, _, segmentation_crops = test.sliding_window(window_size, pad_h, pad_v, threshold)
        # ds_test = ds.crops_to_dataset(crops, labels[:, :, 0], shuffle=False)
        # classif_model = ds.classification_model(img_shape)
        # classif_model.load_weights('weights/classif/classif')
        # test.calssification_predict(classif_model, ds_test, labels[:, :, 0].shape, threshold)
        # var_time = timing('classification prediction', var_time)
        # plt.savefig(img_dir+'test_classif')


if __name__ == "__main__":
    main()
