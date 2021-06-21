import numpy as np
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'figure.dpi' : 150,
          'figure.constrained_layout.use': True}

plt.rcParams.update(params)
import time
import cv2
import matplotlib.patches as patches

def timing(part='', start=None):
    if start!=None:
        print('Time of part '+part+':'+str(time.time()-start))
    return time.time()

class Image:
    """docstring for ."""

    def __init__(self, height, width=640):
        super(Image, self).__init__()
        self.height = height
        self.width = width
        self.create_image()
        self.objects = []
        self.lines = []

    def create_image(self):
        self.image = np.ones((self.height, self.width), np.float32) * 2
        self.image = noisy(self.image, 30, 30)
        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.segmentation = self.mask.copy()
        pass

    def add_object(self, starting_pt, spacing, length, lines):
        times = 4
        big_image = np.zeros((self.height*times,self.width*times), np.float32)

        starting_pt = np.array(starting_pt)*times
        spacing *= times
        length *= times

        self.lines.append(np.zeros((lines,2,2)))
        start = np.array(starting_pt - [length // 2, 0])
        rect_top = tuple(start//times-2)
        end = np.array(starting_pt + [length // 2, 0])

        min_intensity = 50
        for line in range(lines):
            intensity = np.random.randint(min_intensity,200)
            thickness = np.random.randint(times,2*times)
            length_var = [np.random.randint(-times,times),0]
            cv2.line(big_image, tuple(start+length_var), tuple(end-length_var), intensity, thickness)
            self.lines[-1][line][:,:] = np.stack((start+length_var,end-length_var))
            start[1] += spacing
            end[1] += spacing
        self.lines[-1] //= times
        rect_bottom = tuple((end-[0,spacing])//times+2)
        self.mask = cv2.rectangle(self.mask, rect_top, rect_bottom, 255, -1)

        image_resize = cv2.resize(big_image,(self.width,self.height))
        #image_resize = cv2.GaussianBlur(image_resize,(5,5),0.6)
        mask_2 = image_resize.copy().astype(np.uint8)
        _, mask_2 = cv2.threshold(mask_2, min_intensity, 255, cv2.THRESH_BINARY)
        self.segmentation += mask_2
        mask_2 = cv2.bitwise_not(mask_2)

        self.image = cv2.bitwise_and(self.image, self.image, mask = mask_2)
        self.image = cv2.add(self.image, image_resize)
        self.objects.append({'spacing': spacing//times, 'length': length//times, 'lines': lines, 'coords': (rect_top,rect_bottom)})
        pass

    def plot_masked(self):
        masked = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        fig = plt.imshow(masked, vmin=0, vmax=255)
        pass

    def plot_label(self):
        fig = plt.figure()
        plt.imshow(self.image, vmin=0, vmax=255)
        for obj in self.objects:
            coords = obj['coords']
            pt = (coords[0][0],coords[0][1])
            w = coords[1][0] - coords[0][0]
            h = coords[1][1] - coords[0][1]
            plt.gca().add_patch(patches.Rectangle(pt,w,h,linewidth=1,edgecolor='r',facecolor='none'))
            text = ''
            for attribute in list(obj.keys())[:-1]:
                if attribute != list(obj.keys())[-2]:
                    text += attribute + ': '+str(obj[attribute])+'\n'
                else:
                    text += attribute + ': '+str(obj[attribute])
            pad = 5
            plt.gca().text(pt[0]+w+1+pad/2, pt[1]+pad/2, text,
                            color='white',
                            horizontalalignment='left',
                            verticalalignment='top',
                            bbox=dict(facecolor='black', alpha=0.5, pad=pad, linewidth=0))

        return fig

    def sliding_window(self, window_size, pad_h, pad_v):
        windows_h = (self.width - window_size) // pad_h + 1
        windows_v = (self.height - window_size) // pad_v + 1
        crops = np.zeros((windows_v, windows_h, window_size, window_size))
        labels = np.zeros((windows_v, windows_h))
        segmentation_crops = np.zeros((windows_v, windows_h, window_size, window_size))
        number = np.zeros_like(labels)
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
                if labels[j,i] :
                    lower = ([x_top,y_top]<self.lines[-1]).any(axis=1).all(axis=1)
                    higher = (self.lines[-1]<[x_bottom,y_bottom]).any(axis=1).all(axis=1)
                    slice = np.logical_and(lower,higher)
                    number[j,i]= self.lines[-1][slice].shape[0]
                    segmentation_crop = self.segmentation[y_top:y_bottom, x_top:x_bottom]
                    segmentation_crops[j,i,:,:] = segmentation_crop

        return crops, labels, number, segmentation_crops

    def compare_labels(self, resampled_labels, pad_h, pad_v, window_size):
        new_labels = self.resize_labels(resampled_labels, pad_h, pad_v, window_size)

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

        # pairs = np.array(np.where(resampled_labels>0)).transpose()[:,[1, 0]]
        # new_points = np.array([element*[pad_h,pad_v]+[window_size//2, window_size//2] for element in pairs ])
        # plt.scatter(new_points[:,0],new_points[:,1], s=1)
        # And a corresponding grid
        ax.grid(which='both')
        #ax.grid(which='minor', alpha=0.5, color='black')
        #ax.grid(which='major', alpha=0.5, color='black')
        pass

    def resize_labels(self, labels, pad_h, pad_v, window_size):
        labels_resize = np.zeros_like(self.mask, np.float32)

        obj = np.array(np.where(labels>0))
        pairs = obj.transpose()[:,[1, 0]]
        #labels *= 255
        # print(labels.max())
        for pair in pairs:
            start = tuple(pair*[pad_h,pad_v])
            end = tuple(pair*[pad_h,pad_v] + [window_size, window_size])
            #pair = tuple(pair)
            pair = (pair[1], pair[0])
            labels_resize = cv2.rectangle(labels_resize, start, end, 1, -1)

        return labels_resize


def noisy(image, height, intensity):
    row,col= image.shape
    out = np.copy(image)
    # s_vs_p = 0.5
    # amount = 0.1
    # # Salt mode
    # num_salt = np.ceil(amount * image.size * s_vs_p)
    # coords = [np.random.randint(0, i - 1, int(num_salt))
    #       for i in image.shape]
    # out[tuple(coords)] += height
    # # Pepper mode
    # num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    # coords = [np.random.randint(0, i - 1, int(num_pepper))
    #       for i in image.shape]
    # out[tuple(coords)] -= height
    random = np.round(np.random.rand(row, col) * intensity)
    out += random
    return out
