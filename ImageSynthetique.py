import matplotlib.patches as patches
import time
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 7.5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'figure.dpi': 300,
          'figure.constrained_layout.use': True}

plt.rcParams.update(params)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def format_text(object, key, new_line=True):
    """Format text from object caracteristics."""
    formatted = f"{key}: {object[key]}"
    if new_line: formatted += '\n'
    return formatted


def timing(part='', start=None):
    """Time code sections.

    args:
        part - str
        start - timestamp
    """
    if start is not None:
        elapsed = time.time() - start
        elapsed_min = int(np.floor(elapsed)) // 60
        elapsed_sec = elapsed - elapsed_min*60
        print(f"{part} took {elapsed_min}m{elapsed_sec:.2f}s")
    return time.time()


class ImageSynthetique:
    """Docstring for Image class, with methods for generating a custom image.

    params:
        height (int): image height
        width (int): image width
        noise_lvl (float): percentage of maximum grayscale value
        seed (int): optional seed that determines random state
    """

    def __init__(self, height, width=640, noise_lvl=0.1, seed=None, classes=1):
        """Initialize Image object with height and width."""
        # super(Image, self).__init__()
        self.height = height
        self.width = width
        self.figsize = ((width+60)/40, height/40)
        self.noise_lvl = noise_lvl
        self.seed = seed
        self.classes = classes
        self._create_image(self.classes)

    def _create_image(self, classes):
        """Generate canvas for image.

        params:
            noise_lvl (float): percentage of maximum grayscale value
            seed (int): optional seed that determines random state
        """
        self.image = np.zeros((self.height, self.width), np.float32)
        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.segmentation = np.zeros(self.mask.shape+(self.classes,))
        self.mask = np.zeros((self.height, self.width, self.classes), np.uint8)
        self.segmentation = np.zeros_like(self.mask)
        self.objects, self.lines = [], []
        self.predicted, self.labels = {}, {}
        self.finished = False
        self.sliding = False

    def clear(self):
        self._create_image(self.classes)

    def _noisy(self, intensity, seed):
        """Add noise to blank image.

        params:
            intensity (int): grayscale value
            seed (int): optional seed that determines random state
        """
        np.random.seed(seed)
        random = np.random.normal(loc=intensity,
                                scale=intensity/2,
                                size=(self.height, self.width)).round()
        #self.image += random
        return np.clip(self.image + random, 0, 255)

    def _signaltonoise_dB(self, axis=-1, ddof=0):
        m = self.image.mean()
        sd = self.noise_lvl/2*255
        return 20*np.log10(m/sd)

    def finish(self):
        self.image = self._noisy(self.noise_lvl*255, self.seed)
        self.finished = True
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
        rect_top = tuple(start//times-[l_var, 2])
        end = np.array(starting_pt + [length // 2, 0])

        min_intensity = 50
        rng = np.random.RandomState(seed)
        for line in range(lines):
            intensity = rng.randint(min_intensity, (0.8*255))
            thickness = rng.randint(times, 2*times)
            length_var = [rng.randint(-l_var*times, l_var*times), 0]
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
        rect_bottom = tuple((end-[0, spacing])//times+[l_var, 2])
        self.mask[:,:,0] = cv2.rectangle(self.mask[0], rect_top, rect_bottom, 255, -1)

        image_resize = cv2.resize(big_image, (self.width, self.height))
        segmentation = image_resize.copy().astype(np.uint8)
        _, segmentation = cv2.threshold(segmentation, min_intensity-1, 255, cv2.THRESH_BINARY)
        self.segmentation[:, :, 0] += segmentation

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


    def add_lines(self, starting_pt, spacing, spacing_var, thickness, lines, length=10, l_var=1, seed=None):
        """Add line object to image canvas.

        params:
            starting_pt (array_like): point from which to start lines
            spacing  (int): space inbetween lines
            thickness (int): maximum thickness (random)
            length (int): maximum line length (random)
            l_var (int): maximum length variation (random)
            lines (int): number of lines
            seed (int): optional seed that determines random state
        """
        self.objects.append({'type': 'line',
                            'spacing': spacing,
                            'spacing_var': int(spacing_var*spacing),
                            'thickness': thickness,
                            'length': length,
                            'length_var': l_var,
                            'lines': lines})
        starting_pt = np.array(starting_pt)
        rect_top = tuple((starting_pt - [length // 2, 0] - [l_var, thickness]))

        times = 4
        big_image = np.zeros((self.height*times, self.width*times), np.float32)

        starting_pt *= times
        spacing *= times
        length *= times
        center = np.array(starting_pt)

        min_intensity = 50
        rng = np.random.RandomState(seed)
        for line in range(lines):
            intensity = rng.randint(min_intensity, 220)
            thick = rng.randint(times//2, thickness*times//2)
            length_var = rng.randint(-l_var*times, l_var*times)
            # Create lines
            cv2.ellipse(big_image, tuple(center), ((length+length_var)//2, thick),
                    0, 0, 360, intensity, -1)
            if line != (lines-1):
                center[1] += spacing * (rng.random() + spacing_var)

        rect_bottom = tuple((np.array(center) + [length // 2, 0])//times + [l_var, thickness])
        self.objects[-1]['coords'] = (rect_top, rect_bottom)
        self.mask[2] = cv2.rectangle(self.mask[2], rect_top, rect_bottom, 255, -1)

        image_resize = cv2.resize(big_image, (self.width, self.height))
        segmentation = image_resize.copy().astype(np.uint8)
        _, segmentation = cv2.threshold(segmentation, min_intensity-1, 255, cv2.THRESH_BINARY)
        self.segmentation[:, :, 2] += segmentation
        self.image = cv2.add(self.image, image_resize)


    def add_vline(self, starting_pt, length, intensity, thickness=10):
        self.objects.append({'type': 'vline',
                            'starting point': starting_pt,
                            'intensity': intensity,
                            'length': length,
                            'thickness': thickness})
        times = 4
        big_image = np.zeros((self.height*times, self.width*times), np.float32)

        starting_pt = np.array(starting_pt)
        self.objects[-1]['coords'] = (tuple(starting_pt-thickness), tuple(starting_pt+thickness+[0, length]))
        starting_pt *= times
        length *= times
        thickness *= times

        big_image = cv2.line(big_image, starting_pt, starting_pt+[0, length], intensity, thickness=thickness)

        image_resize = cv2.resize(big_image, (self.width, self.height))
        image_resize = cv2.GaussianBlur(image_resize, (7, 7), 50)
        self.image = cv2.add(self.image, image_resize)


    def show(self, with_coords=True):
        """Plot image with labels, shows rectangle around objects and caracteristics.

         params:
            with_coords (bool): option to print out minimum and maximum coordinates.
        """
        if self.finished == False:
            self.finish()

        plt.figure(figsize=(self.figsize))
        plt.title('Image synthétique avec labels et paramètres, SNR=%.2fdB' % self._signaltonoise_dB())
        plt.imshow(self.image, vmin=0, vmax=255)
        for o in self.objects:
            coords = o['coords']
            pt = (coords[0][0], coords[0][1])
            w = coords[1][0] - coords[0][0]
            h = coords[1][1] - coords[0][1]
            # Display type
            text = f"type: {o['type']}\n"
            v_align = 'top'
            if o['type'] == 'ladder':
                if with_coords:
                    # Display minimum and maximum vertical coordinates
                    text += 'y: ({}:{})\n'.format(o['coords'][0][1], o['coords'][1][1])
                # Display spacing in pixels
                text += format_text(o, 'spacing')
                # Display length and max variation in length
                text += 'length: {}±{}\n'.format(o['length'], o['length_var'])
                # Number of lines
                text += format_text(o, 'lines', new_line=False)
            elif o['type'] == 'disk':
                # Display center coordinates
                text += 'center: ({}, {})\n'.format(o['center'][1], o['center'][0])
                # Display diameter
                text += format_text(o, 'diameter', new_line=False)
                v_align = 'center'
                pt = (pt[0]-2, pt[1]-2)
                w += 2
                h += 2
            elif o['type'] == 'line':
                if with_coords:
                    # Display minimum and maximum vertical coordinates
                    text += 'y: ({}:{})\n'.format(o['coords'][0][1], o['coords'][1][1])
                # Display spacing in pixels
                text += 'spacing: {}±{}\n'.format(o['spacing'], o['spacing_var'])
                # Display line thickness
                text += format_text(o, 'thickness')
                # Display length and max variation in length
                text += 'length: {}±{}\n'.format(o['length'], o['length_var'])
                # Number of lines
                text += format_text(o, 'lines', new_line=False)
            elif o['type'] == 'vline':
                if with_coords:
                    # Display minimum and maximum vertical coordinates
                    text += 'y: ({}:{})\n'.format(o['starting point'][1], o['starting point'][1]+o['length'])
                # Display line intensity
                text += format_text(o, 'intensity')
                # Display length
                text += format_text(o, 'length')
                # Display line thickness
                text += format_text(o, 'thickness', new_line=False)
            plt.gca().add_patch(patches.Rectangle(pt, w, h,
                                                linewidth=1, edgecolor='r',
                                                facecolor='none'))
            pad = 5
            plt.gca().text(pt[0]+w+1+pad/2, pt[1]+pad/2, text,
                            color='white',
                            horizontalalignment='left',
                            verticalalignment=v_align,
                            bbox=dict(facecolor='black', alpha=0.5, pad=pad, linewidth=0))

        plt.colorbar(shrink=0.9, pad=0.05, aspect=30, anchor=(0,0.9))
        pass


    def sliding_window(self, window_size, step_h, step_v, network='classif'):
        self.sliding = True
        self.window_size = window_size
        self.step_h, self.step_v = (step_h, step_v)
        windows_h = (self.width - window_size) // step_h +1
        windows_v = (self.height - window_size) // step_v +1
        # self.crops = np.zeros((windows_v, windows_h, window_size, window_size, 1))
        # self.labels['segm'] = np.zeros((windows_v, windows_h, window_size, window_size, self.classes+1))
        maximum = self.mask.max()
        self.crops = view_as_windows(self.image, window_size, step=(step_h,step_v))
        self.labels['segm'] = view_as_windows(self.segmentation, (window_size, window_size, self.classes), step=(step_h,step_v, 1)).squeeze()
        self.labels['segm'] = np.expand_dims(self.labels['segm'], axis=-1)

        self.labels['classif'] = (view_as_windows(self.mask, (window_size, window_size, self.classes), step=(step_h,step_v, 1)).squeeze()>0)
        self.labels['classif'] = self.labels['classif'].any(axis=(-2,-1)).astype(int)
        crops = self.crops.reshape(-1, 32, 32, 1)
        if network == 'classif':
            classif = keras.utils.to_categorical(self.labels['classif'].reshape(-1), self.classes+1)
            return crops, classif
        elif network == 'segm':
            return crops, slef.labels['segm']

    def crops_to_dataset(self, batch_size=32, network='classif', step=1, balanced=False, split=False, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(self.sliding_window(32, step, step, network=network))
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(ds_size)

        if split:
            split = {'train': 0.8, 'val': 0.2}

            for i in split:
                split[i] = int(split[i] * ds_size)

            ds_train = ds.take(split['train'])
            ds_val = ds.take(split['val'])

            ds_train = ds_train.prefetch(batch_size)
            ds_val = ds_val.prefetch(batch_size//2)
            return ds_train, ds_val
        else:
            return ds.prefetch(batch_size)


    def compare_labels(self, resampled_labels, threshold):
        self.predicted['classif'] = self._resize_labels(resampled_labels)

        fig = plt.figure(figsize=(self.figsize))
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
        ax.set_xticks(np.arange(0, self.width, self.step_h), minor=True)
        ax.set_yticks(np.arange(0, self.height, self.step_v), minor=True)
        ax.set_xticks(np.arange(0, self.width, self.step_h*4))
        ax.set_yticks(np.arange(0, self.height, self.step_v*4))
        plt.title('Zones labellisées, avec un seuil de '+str(threshold))
        plt.imshow(self.mask, vmin=0, vmax=255)
        plt.imshow(self.predicted['classif'], vmin=0, vmax=1, alpha=0.5)

        # pairs = np.array(np.where(resampled_labels>0)).transpose()[:,[1, 0]]
        # new_points = np.array([element*[self.step_h,self.step_v]+[self.window_size//2, self.window_size//2] for element in pairs ])
        # plt.scatter(new_points[:,0],new_points[:,1], s=1)
        # And a corresponding grid
        ax.grid(which='both')
        # ax.grid(which='minor', alpha=0.5, color='black')
        # ax.grid(which='major', alpha=0.5, color='black')
        pass

    def _resize_labels(self, labels):
        labels_resize = np.zeros_like(self.mask, np.float32)

        obj = np.array(np.where(labels>0))
        pairs = obj.transpose()[:,[1, 0]]
        for pair in pairs:
            start = tuple((pair-1)*[self.step_h,self.step_v]+self.window_size//2+[self.step_h//2,self.step_v//2])
            end = tuple((pair)*[self.step_h,self.step_v]+self.window_size//2+[self.step_h//2,self.step_v//2])
            pair = (pair[1], pair[0])
            labels_resize = cv2.rectangle(labels_resize, start, end, 1, -1)

        return labels_resize

    def classification_predict(self, model, threshold):
        y_pred = model.predict(self.crops.reshape(-1,32,32,1))
        y_pred = np.reshape(y_pred, self.crops.shape[:2]+(2,))
        indexes = np.where(y_pred>threshold)[:2]
        classified_crops = np.expand_dims(self.crops[indexes], axis=-1)
        mask = np.argmax(y_pred, axis=-1)
        unsupervised_labels = np.argmax(y_pred[indexes], axis=-1)
        mask[indexes] = unsupervised_labels + 2
        plt.figure(figsize=self.figsize)
        plt.title('Prédiction du réseau de classification')
        cmap = {0:[0.8,0.8,1.0,1], 1:[1,0.4,1,1], 2:[0.3,0.3,1.0,1], 3:[0.5,0.1,0.3,1]}
        labels = {0:'Fond (faible)', 1: 'Échelle (faible)', 2:'Fond détecté', 3:' Échelle détectée'}
        arrayShow = np.array([[cmap[i] for i in j] for j in mask])
        patches_ =[patches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
        plt.imshow(arrayShow)
        plt.legend(handles=patches_, loc=4, borderaxespad=0.)
        # plt.figure(figsize=(20, 5))
        # plt.subplot(121)
        for obj in self.objects:
            coords = obj['coords']
            pt = (coords[0][0]/self.step_h, coords[0][1]/self.step_v)
            h = coords[1][1] - coords[0][1]
            h /= self.step_v
            if obj['type'] == 'ladder':
                text = 'spacing: '+str(obj['spacing'])
            elif obj['type'] == 'disk':
                text = 'diameter: ' + str(obj['diameter'])
            plt.gca().text(pt[0], pt[1]+h, text,
                                    color='white',
                                    horizontalalignment='center',
                                    verticalalignment='top',
                                    bbox=dict(facecolor='black', alpha=0.5, linewidth=0))

        # plt.imshow(predict, vmin=0, vmax=1)
        plt.savefig('images/test_train_classif')
        plt.show()
        # plt.subplot(122)
        # plt.imshow(self.predicted['classif'], vmin=0, vmax=1)
        # plt.yticks([])
        return classified_crops, unsupervised_labels


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
        plt.figure(figsize=self.figsize)
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
        darkred = np.array([237, 76, 2])/255.
        lightred = np.array([255, 151, 15])/255.
        darkgreen = np.array([0, 34, 147])/255.

        cmap = {0: lightgreen, 1: lightred, 2: darkred, 3: darkgreen}
        labels = {0: 'Vrai négatif', 2: 'Faux négatif', 1: 'Faux positif', 3: 'Vrai positif'}
        arrayShow = np.array([[cmap[i] for i in j] for j in img])
        # create patches as legend
        patches_ =[patches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
        # plt.imshow(segm)
        plt.imshow(arrayShow)
        plt.legend(handles=patches_, loc=4, borderaxespad=0.)

        ax.grid(which='both', color='black')
        pass

    def compare_predicted(self):
        fig = plt.figure(figsize=self.figsize)
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

        ax.set_xticks(np.arange(0, self.width, self.step_h))
        ax.set_yticks(np.arange(0, self.height, self.step_v))
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
        print(cf_matrix)
        sensibilite = cf_matrix[1, 1] / (cf_matrix[1, 1] + cf_matrix[1, 0])
        specificite = cf_matrix[0, 0] / (cf_matrix[0, 0] + cf_matrix[0, 1])
        print('Sensibilité : '+str(sensibilite))
        print('Spécificité : '+str(specificite))
        return cf_matrix, sensibilite, specificite

    def reshape_labels(self):
        shp = self.labels['classif'].shape
        classif = self.labels['classif'].reshape(shp[0]*shp[1])
        classif = tf.one_hot(classif, self.classes+1)
        shp = self.labels['segm'].shape
        segm = self.labels['segm'].reshape(shp[0]*shp[1], shp[2], shp[3], shp[4])
        return [classif, segm]

    def test_train(self, classification_model, segmentation_model, learn=False, threshold=0.9):
        if not self.sliding:
            self.sliding_window(32,32,32)
        elif self.step_h != 32:
            self.sliding_window(32,32,32)
        var_time = timing()
        classified_crops, unsupervised_labels = self.classification_predict(classification_model, threshold)
        print('Test avec le modèle classification')
        print('Exemples avec une réponse forte: {}'.format(classified_crops.shape[0]))
        if learn:
            print('Updating network with new examples')
            unsupervised_labels = keras.utils.to_categorical(unsupervised_labels, self.classes+1)
            classification_model.fit(classified_crops, unsupervised_labels)
            var_time = timing('classification training', var_time)

        # print('Training segmentation model')
        #plt.imshow(datagen_s.mask)
        var_time = timing('segmentation training')
        results = segmentation_model.evaluate(self.crops.reshape(-1,32,32,1), self.labels['segm'].reshape(-1,32,32,1)/255, verbose=0)
        y_pred = segmentation_model.predict(self.crops.reshape(-1,32,32,1))
        mask = y_pred
        # mask = np.zeros_like(y_pred)
        # mask[np.where(y_pred > threshold)] +=1
        # print(mask.shape)
        mask = mask.reshape(self.height//32, self.width//32, 32, 32, 1)
        mask = mask.swapaxes(1, 2)
        mask = mask.reshape((self.height//32)*32, (self.width//32)*32,1)
        plt.title('Prédiction du réseau de sesegmentation')
        plt.imshow(mask[:,:,0])
        text = ''
        for i,name in enumerate(segmentation_model.metrics_names):
            if i != 0:
                text += f"{name}: {results[i]:.4f}\n"
        text = text[:-2]
        pad = 5
        plt.gca().text(pad*2, self.height-100+pad/2, text,
                        color='white',
                        horizontalalignment='left',
                        verticalalignment='center',
                        bbox=dict(facecolor='black', alpha=0.5, pad=pad, linewidth=0))
        plt.savefig('images/test_train_segm', dpi=300)
        plt.show()

def reshape_dataset(dataset):
    shp = dataset.shape
    return dataset.reshape((-1,) + shp[2:])
