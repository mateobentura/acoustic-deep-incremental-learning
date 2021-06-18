from ImageSynthetique import *
#from Dataset import *


def main():
    height, width = (300, 640)

    var_time = timing()
    # Image test
    train = Image(height, width)
    train.add_object(starting_pt = [50, 40],
                    spacing = 7, length = 12,lines = 15)
    train.add_object(starting_pt = [150, 80],
                    spacing = 12, length = 14,lines = 15)
    train.plot_label()
    plt.savefig('train')
    var_time = timing('train', var_time)

    # Image test
    test = Image(300)
    # Premier objet
    test.add_object(starting_pt = [50, 20],
                    spacing = 10, length = 10, lines = 8)
    # Deuxième objet
    test.add_object(starting_pt = [350, 200],
                    spacing = 6, length = 5, lines = 10)

    fig = test.plot_label()
    plt.savefig('test')
    var_time = timing('test', var_time)

    # Sliding window
    window_size = 32
    pad_h = 16
    pad_v = 4

    crops, labels, number = train.sliding_window(window_size, pad_h, pad_v)
    var_time = timing('sliding_window', var_time)

    plt.imshow(crops[crops.shape[0]//2-2,crops.shape[1]//2-1])
    plt.imshow(labels)
    plt.savefig('crop')

    train.compare_labels(labels, pad_h, pad_v, window_size)
    var_time = timing('compare', var_time)
    plt.savefig('compare')
    var_time = timing('save', var_time)

    labels /= labels.max()
    img_shape = (window_size,window_size,1)
    # ds_train, ds_val = crops_to_dataset(crops, labels, balanced=False, split=True)

if __name__ == "__main__":
    main()
