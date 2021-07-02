from ImageSynthetique import *
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from Dataset import *
import sys

def main():
    height, width = (300, 640)
    threshold = float(sys.argv[2])
    var_time = timing()
    if sys.argv[1] == 'train':
        # Image test
        train = Image(height, width)
        train.add_ladder(starting_pt = [24*4, 25],
                        spacing = 7, length = 12, l_var =1, lines = 32)
        train.add_ladder(starting_pt = [350, 30],
                        spacing = 13, length = 12, l_var =1, lines = 18)
        train.add_star(center = [200, 200], radius=3, intensity=100)
        train.plot_label()
        #plt.imshow(train.image)
        plt.savefig('train')
        var_time = timing('train', var_time)

        # Sliding window
        window_size = 32
        pad_h = 1
        pad_v = 1

        crops, labels, number, segmentation_crops = train.sliding_window(window_size, pad_h, pad_v, threshold)
        var_time = timing('sliding_window', var_time)

        indexes = (40//pad_v,50//pad_h-1)
        plt.subplot(121)
        plt.imshow(crops[indexes])
        plt.subplot(122)
        plt.imshow(segmentation_crops[indexes])
        plt.savefig('crop')

        train.compare_labels(labels, threshold)
        var_time = timing('compare', var_time)
        plt.savefig('compare')
        var_time = timing('save', var_time)

        labels /= labels.max()
        img_shape = (window_size,window_size,1)
        # ds_train, ds_val = crops_to_dataset(crops, labels, balanced=False, split=True)
    if sys.argv[1] == 'test':
        # Image test
        test = Image(300)
        # Premier objet
        spacings = [5, 7, 9, 11, 13, 15]
        for spacing in spacings:
            pt = [47*(spacing-3), 30]
            test.add_ladder(starting_pt = pt,
                        spacing = spacing, length = 12, l_var=2, lines = 4*(55//spacing))

        fig = test.plot_label()
        plt.savefig('test')
        var_time = timing('test', var_time)

if __name__ == "__main__":
    main()
