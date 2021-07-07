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
          'figure.dpi': 300,
          'figure.constrained_layout.use': True}

plt.rcParams.update(params)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def format_text(object, key, new_line=True):
    """Format text from object caracteristics."""
    formatted = f"{key}: {object[key]}"
    if new_line: formatted += '\n'
    return formatted


class Image:
    """Docstring for Image class, with methods for generating a custom image.

    params:
        height (int): image height
        width (int): image width
        noise_lvl (float): percentage of maximum grayscale value
        seed (int): optional seed that determines random state
    """

    def __init__(self, height, width=640, noise_lvl=0.3, seed=None):
        """Initialize Image object with height and width."""
        # super(Image, self).__init__()
        self.height = height
        self.width = width
        self.create_image(noise_lvl, seed)
        self.objects = []
        self.lines = []
        self.predicted = {}

    def create_image(self, noise_lvl, seed, classes=3):
        """Generate canvas for image.

        params:
            noise_lvl (float): percentage of maximum grayscale value
            seed (int): optional seed that determines random state
        """
        self.image = np.ones((self.height, self.width), np.float32) * 2
        self.noisy(noise_lvl*255, seed)
        self.classes = classes
        self.mask = np.zeros((self.height, self.width, self.classes), np.uint8)
        self.segmentation = np.zeros_like(self.mask)
        pass

    def clip(self):
        """Clip image to maximum (255)."""
        self.image = np.clip(self.image, 0, 255)
        pass

    def noisy(self, intensity, seed):
        """Add noise to blank image.

        params:
            intensity (int): grayscale value
            seed (int): optional seed that determines random state
        """
        np.random.seed(seed)
        random = np.random.normal(loc=intensity / 2,
                                scale=intensity / 2,
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
        rect_top = tuple(start//times-[l_var, 2])
        end = np.array(starting_pt + [length // 2, 0])

        min_intensity = 50
        rng = np.random.RandomState(seed)
        for line in range(lines):
            intensity = rng.randint(min_intensity, 220)
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
        self.mask[0] = cv2.rectangle(self.mask[0], rect_top, rect_bottom, 255, -1)

        image_resize = cv2.resize(big_image, (self.width, self.height))
        segmentation = image_resize.copy().astype(np.uint8)
        _, segmentation = cv2.threshold(segmentation, min_intensity-1, 255, cv2.THRESH_BINARY)
        self.segmentation[:, :, 0] += segmentation

        self.image = cv2.add(self.image, image_resize)
        self.clip()
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
        self.clip()
        pass

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
        self.clip()
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
            # Display type
            text = f"type: {o['type']}\n"
            if o['type'] == 'ladder':
                if with_coords:
                    # Display minimum and maximum vertical coordinates
                    text += 'y: ({}:{})\n'.format(*o['coords'][:][1])
                # Display spacing in pixels
                text += format_text(o, 'spacing')
                # Display length and max variation in length
                text += 'length: {}±{}\n'.format(o['length'], o['length_var'])
                # Number of lines
                text += format_text(o, 'lines', new_line=False)
                v_align = 'top'
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
                v_align = 'top'

            plt.gca().add_patch(patches.Rectangle(pt, w, h,
                                                linewidth=1, edgecolor='r',
                                                facecolor='none'))
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
        crops = np.zeros((windows_v, windows_h, window_size, window_size))
        labels = np.zeros((windows_v, windows_h, self.classes))
        segmentation_crops = np.zeros((windows_v, windows_h, window_size, window_size, self.classes))
        number = np.zeros_like(labels)
        max = self.mask.max()
        for j in range(windows_v):
            y_m = j*pad_v
            y_p = j*pad_v + window_size
            for i in range(windows_h):
                x_m = i*pad_h
                x_p = i*pad_h + window_size
                crop = self.image[y_m:y_p, x_m:x_p]
                crops[j, i, :, :] = crop
                mask_crop = self.mask[y_m:y_p, x_m:x_p]
                # if mask_crop[mask_crop>0].size > threshold:
                #     labels[j,i] = 1
                # labels[j,i] = mask_crop[mask_crop>0].size
                if mask_crop[mask_crop > 0].size:
                    segmentation_crop = self.segmentation[y_m:y_p, x_m:x_p]
                    if mask_crop[mask_crop > 0].size > threshold * max:
                        labels[j, i] = 1
                    segmentation_crops[j, i] = segmentation_crop

        return crops, labels, number, segmentation_crops

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
        predict_thr = np.where(predict > threshold, 1, 0)
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
        plt.imshow(predict_thr, vmin=0, vmax=1)
        plt.yticks([])
        return predict_thr



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
        plt.imshow(self.predicted['segm'])
        plt.imshow(self.segmentation, alpha=0.5)
        plt.xticks(np.arange(0,self.width, 32))
        plt.yticks(np.arange(0,self.height, 32))
        plt.grid(color='black')
        #plt.show()
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

        if type == 'segm':
            mask = self.segmentation.reshape(-1)/255
        elif type == 'classif':
            mask = self.mask.reshape(-1)/255

        cf_matrix = tf.math.confusion_matrix(mask, predicted).numpy()
        group_names = ["Vrai fond", "Fausse échelle", "Faux fond", "Vraie échelle"]
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_counts[-1] += ' ('+str(sum(cf_matrix[1, :]))+')'
        cf_labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]

        cf_labels = np.asarray(cf_labels).reshape(2, 2)

        sns.heatmap(cf_matrix,
                    annot=cf_labels, fmt="",
                    cmap=['lightgray'], cbar=False,
                    linewidths=0.5, linecolor='black',
                    square=True,
                    xticklabels=['Fond', 'Échelle'], yticklabels=['Fond', 'Échelle'])

        plt.ylabel('Label réel')
        plt.xlabel('Label prédit')
        sensibilite = cf_matrix[1, 1] / (cf_matrix[1, 1] + cf_matrix[0, 1])
        specificite = cf_matrix[0, 0] / (cf_matrix[0, 0] + cf_matrix[1, 0])
        print('Sensibilité : '+str(sensibilite))
        print('Specificité : '+str(specificite))
        return cf_matrix, sensibilite, specificite


def reshape_dataset(dataset):
    shape = dataset.shape
    return dataset.reshape(shape[0]*shape[1], shape[2], shape[3])
