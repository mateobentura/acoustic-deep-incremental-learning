from ImageSynthetique import *

def main():
    height, width = (300, 640)

    train = Image(height, width)

    starting_point = [300, 20]
    spacing = 15
    length = 12
    lines = 15

    train.add_object(starting_point, spacing, length, lines)
    train.plot_label()
    plt.savefig('train')

    # Image test
    test = Image(300)

    # Premier objet
    starting_point = [150, 20]
    spacing = 10
    length = 10
    lines = 8
    test.add_object(starting_point, spacing, length, lines)

    # Deuxième objet
    starting_point = [350, 200]
    spacing = 6
    length = 5
    lines = 8
    test.add_object(starting_point, spacing, length, lines)

    fig = test.plot_label()
    plt.savefig('test')

    pad_h = 8
    pad_v = 4
    window_size = 32

    crops, labels = train.sliding_window(window_size, pad_h, pad_v)

    plt.imshow(crops[crops.shape[0]//2-2,crops.shape[1]//2-1])
    labels /= labels.max()
    plt.imshow(labels)
    plt.savefig('crop')

    compare_labels(train.mask, labels, pad_h, pad_v, window_size, width, height)
    plt.savefig('compare')

if __name__ == "__main__":
    main()
