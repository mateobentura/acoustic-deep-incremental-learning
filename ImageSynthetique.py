import matplotlib.patches as patches
import seaborn as sns
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 7.5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'figure.dpi': 150,
          'figure.constrained_layout.use': True}

plt.rcParams.update(params)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Image:
    """Docstring for Image class, with methods for generating a custom image.

    params:
        height (int): image height
        width (int): image width
        noise_lvl (float): percentage of maximum grayscale value
        seed (int): optional seed that determines random state
    """

    def __init__(self, height, width=640, noise_lvl=0.3, seed=None, classes=1):
        """Initialize Image object with height and width."""
        # super(Image, self).__init__()
        self.height = height
        self.width = width
        self.create_image(noise_lvl, seed, classes)
        self.objects = []
        self.lines = []
        self.predicted = {}
        self.labels = {}

    def create_image(self, noise_lvl, seed, classes):
        """Generate canvas for image.

        params:
            noise_lvl (float): percentage of maximum grayscale value
            seed (int): optional seed that determines random state
        """
        self.image = np.zeros((self.height, self.width), np.float32)
        self.noisy(noise_lvl*255, seed)
        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.segmentation = np.zeros(self.mask.shape+(classes,))
        self.classes = classes
        pass

    def clip(self):
        """Limit image to [0, 255]."""
        self.image = np.clip(self.image, 0, 255)
        pass

    def noisy(self, intensity, seed):
        """Add noise to blank image.

        params:
            intensity (int): grayscale value
            seed (int): optional seed that determines random state
        """
        np.random.seed(seed)
        random = np.random.normal(loc=0,
                                scale=intensity,
                                size=(self.height, self.width)).round()
        self.image += random
        pass

    def add_ladder(self, starting_pt, spacing, length, l_var, lines, seed=None):
        """Add ladder object to image canvas.

        params:
            starting_pt (array_like): point from which to start ladder
            spacing  (int): space inbetween bars of the ladder
            length (int): bar length
            l_var (int): maximum length variation (random)
            lines (int): number of lines
            seed (int): optional seed that determines random state
        """
        times = 4
        big_image = np.zeros((self.height*times, self.width*times), np.float32)

        starting_pt = np.array(starting_pt)*times
        spacing *= times
        length *= times

        self.lines.append(np.zeros((lines, 2, 2)))
        start = np.array(starting_pt - [length // 2, 0])
        rect_top = tuple(start//times-2)
        end = np.array(starting_pt + [length // 2, 0])

        min_intensity = 50
        if seed is not None:
            np.random.seed(seed)
            seeds = np.random.randint(1024, size=lines)
        for line in range(lines):
            if seed is not None: np.random.seed(seeds[line])
            intensity = np.random.randint(min_intensity, 180)
            if seed is not None: np.random.seed(seeds[line])
            thickness = np.random.randint(times, 2*times)
            if seed is not None: np.random.seed(seeds[line])
            length_var = [np.random.randint(-l_var*times, l_var*times), 0]
            # Create bars
            cv2.line(big_image,
                    tuple(start + length_var), tuple(end - length_var),
                    intensity, thickness)
            # Store line coordinates
            self.lines[-1][line][:, :] = np.stack((start + length_var,
                                                    end - length_var))
            start[1] += spacing
            end[1] += spacing
        self.lines[-1] //= times
        rect_bottom = tuple((end-[0, spacing])//times+2)
        self.mask = cv2.rectangle(self.mask, rect_top, rect_bottom, 255, -1)

        image_resize = cv2.resize(big_image, (self.width, self.height))
        mask_2 = image_resize.copy().astype(np.uint8)
        _, mask_2 = cv2.threshold(mask_2, min_intensity-1, 255, cv2.THRESH_BINARY)
        self.segmentation[:, :, 0] += mask_2
        # mask_2 = cv2.bitwise_not(mask_2)
        #
        # self.image = cv2.bitwise_and(self.image, self.image, mask = mask_2)
        self.image = cv2.add(self.image, image_resize)
        self.objects.append({'type': 'ladder',
                            'coords': (rect_top, rect_bottom),
                            'spacing': spacing // times,
                            'length': length // times,
                            'length_var': l_var,
                            'lines': lines})
        pass

    def add_disk(self, center, diameter, intensity):
        """Add dosk object to image canvas.

        params:
            center (array_like): coordinates of the circle's center
            diameter (int): diameter in pixels
            intensity (float): maximum intensity in percentage
        """
        times = 4
        intensity = int(intensity*255)
        radius = diameter / 2
        radius_ceil = int(np.ceil(radius))
        radius_floor = int(np.floor(radius))

        top = tuple(np.array(center) - radius_ceil)
        bottom = tuple(np.array(center) + radius_floor)
        coords = (top, bottom)
        self.objects.append({'type': 'disk',
                            'center': center,
                            'diameter': diameter,
                            'coords': coords})
        big_image = np.zeros((2*diameter*times, 2*diameter*times), dtype='float32')
        big_image = cv2.circle(big_image, (diameter*times, diameter*times),
                            diameter*times//2, intensity, -1)
        circle = cv2.resize(big_image, (diameter*2, diameter*2))
        x_m = coords[0][0] - radius_ceil
        x_p = coords[1][0] + radius_floor
        y_m = coords[0][1] - radius_ceil
        y_p = coords[1][1] + radius_floor
        self.image[y_m:y_p, x_m:x_p] = cv2.add(self.image[y_m:y_p, x_m:x_p], circle)
        _, segmentation = cv2.threshold(circle, intensity-1, 255, cv2.THRESH_BINARY)
        self.segmentation[y_m:y_p, x_m:x_p, 1] = segmentation
        pass

    def plot_masked(self):
        """Plot image masked."""
        masked = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        plt.imshow(masked, vmin=0, vmax=255)
        pass

    def plot_label(self, with_coords=True):
        """Plot image with labels, shows rectangle around objects and caracteristics.

         params:
            with_coords (bool): option to print out minimum and maximum coordinates.
        """
        plt.figure()
        plt.imshow(self.image, vmin=0, vmax=255)
        for o in self.objects:
            coords = o['coords']
            pt = (coords[0][0], coords[0][1])
            w = coords[1][0] - coords[0][0]
            h = coords[1][1] - coords[0][1]

            plt.gca().add_patch(patches.Rectangle(pt, w, h,
                                                linewidth=1, edgecolor='r',
                                                facecolor='none'))
            # Display type
            text = f"type: {o['type']}\n"
            if o['type'] == 'ladder':
                if with_coords:
                    # Display minimum and maximum vertical coordinates
                    text += 'y: ({}:{})\n'.format(*o['coords'][:][1])
                # Display spacing in pixels
                text += 'spacing: {} \n'.format(o['spacing'])
                # Display length and max variation in length
                text += 'length: {}±{}\n'.format(o['length'], o['length_var'])
                # Number of lines
                text += 'lines: {}'.format(o['lines'])
                v_align = 'top'
            if o['type'] == 'disk':
                # Display center coordinates
                text += 'center: ({}, {})\n'.format(*o['center'])
                # Display diameter
                text += f"diameter: {o['diameter']}"
                v_align = 'center'
            pad = 5
            plt.gca().text(pt[0]+w+1+pad/2, pt[1]+pad/2, text,
                            color='white',
                            horizontalalignment='left',
                            verticalalignment=v_align,
                            bbox=dict(facecolor='black', alpha=0.5, pad=pad, linewidth=0))
        plt.colorbar(shrink=0.85, pad=0.01)

        pass

    def sliding_window(self, window_size, pad_h, pad_v, threshold):
        self.window_size = window_size
        self.pad_h, self.pad_v = (pad_h, pad_v)
        windows_h = (self.width - window_size) // pad_h +1
        windows_v = (self.height - window_size) // pad_v +1
        self.crops = np.zeros((windows_v, windows_h, window_size, window_size))
        self.labels['classif'] = np.zeros((windows_v, windows_h, self.classes))
        self.labels['segm'] = np.zeros((windows_v, windows_h, window_size, window_size, self.classes))
        max = self.mask.max()
        for j in range(windows_v):
            y_m = j*pad_v
            y_p = j*pad_v + window_size
            for i in range(windows_h):
                x_m = i*pad_h
                x_p = i*pad_h + window_size
                crop = self.image[y_m:y_p, x_m:x_p]
                self.crops[j, i, :, :] = crop
                mask_crop = self.mask[y_m:y_p, x_m:x_p]
                if mask_crop[mask_crop > 0].size:
                    segmentation_crop = self.segmentation[y_m:y_p, x_m:x_p]
                    if mask_crop[mask_crop > 0].size > threshold * max:
                        self.labels['classif'][j, i] = 1
                    self.labels['segm'][j, i] = segmentation_crop
        pass

    def compare_labels(self, resampled_labels, threshold):
        self.predicted['classif'] = self.resize_labels(resampled_labels)

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
        ax.set_xticks(np.arange(0, self.width, self.pad_h), minor=True)
        ax.set_yticks(np.arange(0, self.height, self.pad_v), minor=True)
        ax.set_xticks(np.arange(0, self.width, self.pad_h*4))
        ax.set_yticks(np.arange(0, self.height, self.pad_v*4))
        plt.title('Zones labellisées, avec un seuil de '+str(threshold))
        plt.imshow(self.mask, vmin=0, vmax=255)
        plt.imshow(self.predicted['classif'], vmin=0, vmax=1, alpha=0.5)

        # pairs = np.array(np.where(resampled_labels>0)).transpose()[:,[1, 0]]
        # new_points = np.array([element*[self.pad_h,self.pad_v]+[self.window_size//2, self.window_size//2] for element in pairs ])
        # plt.scatter(new_points[:,0],new_points[:,1], s=1)
        # And a corresponding grid
        ax.grid(which='both')
        # ax.grid(which='minor', alpha=0.5, color='black')
        # ax.grid(which='major', alpha=0.5, color='black')
        pass

    def resize_labels(self, labels):
        labels_resize = np.zeros_like(self.mask, np.float32)

        obj = np.array(np.where(labels>0))
        pairs = obj.transpose()[:,[1, 0]]
        for pair in pairs:
            start = tuple((pair-1)*[self.pad_h,self.pad_v]+self.window_size//2+[self.pad_h//2,self.pad_v//2])
            end = tuple((pair)*[self.pad_h,self.pad_v]+self.window_size//2+[self.pad_h//2,self.pad_v//2])
            pair = (pair[1], pair[0])
            labels_resize = cv2.rectangle(labels_resize, start, end, 1, -1)

        return labels_resize


    def calssification_predict(self, model, ds_test, shape, threshold):
        predicted_labels = model.predict(ds_test.batch(32))
        predict = predicted_labels.reshape(shape)
        self.predicted['classif'] = np.where(predict > threshold, 1, 0)
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        for obj in self.objects:
            coords = obj['coords']
            pt = (coords[0][0]/self.pad_h, coords[0][1]/self.pad_v)
            h = coords[1][1] - coords[0][1]
            h /= self.pad_v
            if obj['type'] == 'ladder':
                text = 'spacing: '+str(obj['spacing'])
            elif obj['type'] == 'disk':
                text = 'diameter: ' + str(obj['diameter'])
            plt.gca().text(pt[0], pt[1]+h, text,
                                    color='white',
                                    horizontalalignment='center',
                                    verticalalignment='top',
                                    bbox=dict(facecolor='black', alpha=0.5, linewidth=0))

        plt.imshow(predict, vmin=0, vmax=1)
        plt.subplot(122)
        plt.imshow(self.predicted['classif'], vmin=0, vmax=1)
        plt.yticks([])
        pass



    def segmentation_predict(self, model, crops, threshold):
        """Function that shows prediction results.
        input:
            model (Keras Model class)
            crops (numpy array of crops)
            threshold (float)
        return: null
        """
        crops_reshape = np.expand_dims(reshape_dataset(crops), axis=-1)
        predicted = model.predict(crops_reshape)
        predicted = predicted.reshape(predicted.shape[:-1])
        predicted = predicted.reshape(crops.shape)
        predicted_image = np.zeros_like(self.image)
        for row in range(predicted.shape[0]):
            y_m = row*self.window_size
            y_p = (row+1)*self.window_size
            for col in range(predicted.shape[1]):
                x_m = col*self.window_size
                x_p = (col+1)*self.window_size
                predicted_image[y_m:y_p,x_m:x_p] = predicted[row,col]
        self.predicted['segm'] = np.where(predicted_image > threshold, 1, 0)
        plt.figure(figsize=(16, 7.5))
        ax = plt.gca()
        # plt.imshow(self.predicted['segm'])
        # plt.imshow(self.segmentation, alpha=0.5)
        plt.xticks(np.arange(0,self.width, 32))
        plt.yticks(np.arange(0,self.height, 32))
        img = np.zeros_like(self.image)

        fp = np.logical_or(np.logical_not(self.predicted['segm']), self.segmentation[:, :, 0]) == 0
        fn = np.logical_or(self.predicted['segm'], np.logical_not(self.segmentation[:, :, 0])) == 0
        tp = np.logical_and(self.predicted['segm'], self.segmentation[:, :, 0]) > 0

        img[fp] = 1.0
        img[fn] = 2.0
        img[tp] = 3.0

        lightgreen = np.array([5, 91, 252])/255.
        darkred = np.array([153, 0, 0])/255.
        darkgreen = np.array([255, 151, 15])/255.
        lightred = np.array([255, 51, 51])/255.

        cmap = {0: lightgreen, 1: lightred, 2: darkred, 3: darkgreen}
        labels = {0: 'Vrai négatif', 2: 'Faux négatif', 1: 'Fausse positif', 3: 'Vraie positif'}
        arrayShow = np.array([[cmap[i] for i in j] for j in img])
        # create patches as legend
        patches_ =[patches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
        # plt.imshow(segm)
        plt.imshow(arrayShow)
        plt.legend(handles=patches_, loc=4, borderaxespad=0.)

        ax.grid(which='both', color='black')
        pass

    def compare_predicted(self):
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

        ax.set_xticks(np.arange(0, self.width, self.pad_h))
        ax.set_yticks(np.arange(0, self.height, self.pad_v))
        img = np.copy(self.predicted['classif'])
        img[self.predicted['segm']>0] = 2.0
        t = 1 ## alpha value
        cmap = {0:[1.,1.0,1.0,t],1:[0.3,0.3,1.0,t],2:[0.5,0.1,0.3,t]}
        labels = {0:'',1:'Masque de classification',2:'Masque de segmentation'}
        arrayShow = np.array([[cmap[i] for i in j] for j in img])
        cmap = {1: cmap[1],2:cmap[2]}
        ## create patches as legend
        patches_ =[patches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
        # plt.imshow(segm)
        plt.imshow(arrayShow)
        plt.legend(handles=patches_, loc=4, borderaxespad=0.)

        ax.grid(which='both')
        pass

    def confusion_matrix(self, type):
        plt.figure(figsize=(4.5,3))
        predicted = self.predicted[type].reshape(-1)
        if type == 'classif':
            labels = self.labels[type].reshape(-1)
        else:
            labels = self.segmentation.reshape(-1)
            labels /= labels.max()
        cf_matrix = tf.math.confusion_matrix(labels, predicted).numpy()
        # group_names = ["Vrai fond", "Fausse échelle", "Faux fond", "Vraie échelle"]
        # group_counts = ["{0:0.0f}".format(value) for value in
        #                 cf_matrix.flatten()]
        # group_counts[-1] += ' ('+str(sum(cf_matrix[1, :]))+')'
        # cf_labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
        #
        # cf_labels = np.asarray(cf_labels).reshape(2, 2)
        #
        # sns.heatmap(cf_matrix,
        #             annot=cf_labels, fmt="",
        #             cmap=['lightgray'], cbar=False,
        #             linewidths=0.5, linecolor='black',
        #             square=True,
        #             xticklabels=['Fond', 'Échelle'], yticklabels=['Fond', 'Échelle'])
        #
        # plt.ylabel('Label réel')
        # plt.xlabel('Label prédit')
        print(cf_matrix)
        sensibilite = cf_matrix[1, 1] / (cf_matrix[1, 1] + cf_matrix[1, 0])
        specificite = cf_matrix[0, 0] / (cf_matrix[0, 0] + cf_matrix[0, 1])
        print('Sensibilité : '+str(sensibilite))
        print('Spécificité : '+str(specificite))
        return cf_matrix, sensibilite, specificite


def reshape_dataset(dataset):
    shape = dataset.shape
    return dataset.reshape(shape[0]*shape[1], shape[2], shape[3])
