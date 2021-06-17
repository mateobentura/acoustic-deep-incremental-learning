from ImageSynthetique import *

def main():
    height = 300
    width = 640

    # for (i,j) in itertools.product(range(3),range(3)):
    #     print(i)
    #
    # print(a)
    # Image test
    train = Image(height, width)
    train.add_object(starting_pt = [50, 40],
                    spacing = 7, length = 12,lines = 15)
    train.add_object(starting_pt = [150, 80],
                    spacing = 12, length = 14,lines = 15)
    train.plot_label()
    plt.savefig('train')

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

    # Sliding window
    window_size = 32
    pad_h = 16
    pad_v = 4

    crops, labels, number = train.sliding_window(window_size, pad_h, pad_v)

    plt.imshow(number)
    plt.savefig('number')
    # plt.imshow(crops[crops.shape[0]//2-2,crops.shape[1]//2-1])
    # labels /= labels.max()
    # plt.imshow(labels)
    # plt.savefig('crop')
    #
    # resize = resize_labels(labels, pad_h, pad_v, window_size, (height,width))
    # compare_labels(train.mask, resize, pad_h, pad_v, height, width)
    # plt.savefig('compare')
    #
    # plt.figure()
    # plt.imshow(train.segmentation)
    # plt.savefig('segmentation')
if __name__ == "__main__":
    main()
