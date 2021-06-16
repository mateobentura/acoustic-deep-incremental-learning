from ImageSynthetique import *
class Image(object):
    """docstring for ."""

    def __init__(self, height, width=640):
        #super(Image, self).__init__()
        self.height = height
        self.width = width
        self.create_image()
        self.objects = []

    def create_image(self):
        self.image = np.ones((self.height, self.width), np.float32) * 2
        self.image = noisy(self.image, 10)
        self.mask = np.zeros((self.height, self.width), np.uint8)
        pass

    def add_object(self, starting_pt, spacing, length, lines):
        times = 4
        big_image = np.zeros((self.height*times,self.width*times), np.float32)

        starting_pt = np.array(starting_pt)*times
        spacing *= times
        length *= times
        lines = 15

        start = np.array(starting_pt - [length // 2, 0])
        rect_top = tuple(start//times-2)
        end = np.array(starting_pt + [length // 2, 0])

        min_intensity = 50
        for line in range(lines):
            intensity = np.random.randint(min_intensity,200)
            thickness = np.random.randint(times,2*times)
            cv2.line(big_image, tuple(start), tuple(end), intensity, thickness)
            start[1] += spacing
            end[1] += spacing

        rect_bottom = tuple((end-[0,spacing])//times+2)
        self.mask = cv2.rectangle(self.mask, rect_top, rect_bottom, 255, -1)

        image_resize = cv2.resize(big_image,(self.width,self.height))
        #image_resize = cv2.GaussianBlur(image_resize,(5,5),0.6)
        mask_2 = image_resize.copy().astype(np.uint8)
        _, mask_2 = cv2.threshold(mask_2, min_intensity, 255, cv2.THRESH_BINARY)
        mask_2 += image_resize.astype(np.uint8)
        mask_2 = cv2.bitwise_not(mask_2)

        self.image = cv2.bitwise_and(self.image, self.image, mask = mask_2)
        self.image = cv2.add(self.image, image_resize)
        self.objects.append((rect_top,rect_bottom))
        pass

    def plot_masked(self):
        masked = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        fig = plt.imshow(masked, vmin=0, vmax=255)
        pass

    def plot_label(self):
        temp = self.image.copy()
        for obj in self.objects:
            temp = cv2.rectangle(temp, obj[0], obj[1], 255, 1)
        fig = plt.imshow(temp, vmin=0, vmax=255)
        return fig

    def sliding_window(self, window_size, pad_h, pad_v):
        windows_h = (self.width - window_size) // pad_h + 1
        windows_v = (self.height - window_size) // pad_v + 1
        crops = np.zeros((windows_v, windows_h, window_size, window_size))
        labels = np.zeros((windows_v, windows_h))
        for j in range(windows_v):
            y_top = j*pad_v
            y_bottom = j*pad_v + window_size
            for i in range(windows_h):
                x_top = i*pad_h
                x_bottom = i*pad_h + window_size
                crop = self.image[y_top:y_bottom, x_top:x_bottom]
                crops[j,i,:,:] = crop
                mask_crop = self.mask[y_top:y_bottom, x_top:x_bottom]
                # if mask_crop[mask_crop>0].size > 510:
                #     labels[j,i] = 1
                labels[j,i] = mask_crop[mask_crop>0].size
        return crops, labels

def main():
    height = 300
    width = 640

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

    resize = resize_labels(labels, pad_h, pad_v, window_size, (height,width))
    compare_labels(train.mask, resize, pad_h, pad_v, height, width)
    plt.savefig('compare')
if __name__ == "__main__":
    main()
