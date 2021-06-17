import numpy as np
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'figure.dpi' : 200}

plt.rcParams.update(params)

import cv2

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

    def compare_labels(self, resampled_labels, pad_h, pad_v, window_size):
        new_labels = resize_labels(resampled_labels, pad_h, pad_v, window_size, (self.height,self.width))
        fig = plt.figure()
        ax = fig.gca()
        ax.tick_params(
            which='major',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False,
            grid_color='black',
            grid_alpha=0.3)
        ax.tick_params(
            which='minor',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            grid_color='black',
            grid_alpha=0.1)
        ax.set_xticks(np.arange(0, self.width, pad_h), minor=True)
        ax.set_yticks(np.arange(0, self.height, pad_v), minor=True)
        ax.set_xticks(np.arange(0, self.width, pad_h*4))
        ax.set_yticks(np.arange(0, self.height, pad_v*4))

        plt.imshow(self.mask, vmin=0, vmax=255)
        plt.imshow(new_labels, vmin=0, vmax=1, alpha=0.5)
        # And a corresponding grid
        ax.grid(which='both')
        #ax.grid(which='minor', alpha=0.5, color='black')
        #ax.grid(which='major', alpha=0.5, color='black')
        pass


def noisy(image, height):
    row,col= image.shape
    s_vs_p = 0.5
    amount = 0.08
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[tuple(coords)] += height
    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    out[tuple(coords)] -= height
    return out


def resize_labels(labels, pad_h, pad_v, window_size, shape):
    labels_resize = np.zeros(shape, dtype='uint8')

    obj = np.array(np.where(labels>0))
    pairs = obj.transpose()[:,[1, 0]]

    for pair in pairs:
        start = tuple(pair*[pad_h,pad_v] + [ window_size,0])
        end = tuple(pair*[pad_h,pad_v] + [0, window_size])
        cv2.rectangle(labels_resize, start, end, 1, -1)
    return labels_resize
