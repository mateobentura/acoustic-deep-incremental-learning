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

def create_background(height, width, background=2):
    image = np.ones((height,width), np.float32) * background
    image = noisy(image, 10)
    mask = np.zeros((height, width), np.uint8)
    return image, mask


def add_bars(image, mask, starting_pt, spacing, length, lines):
    times = 4
    height, width = image.shape
    big_image = np.zeros((height*times,width*times), np.float32)

    starting_pt *= times
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
    mask = cv2.rectangle(mask, rect_top, rect_bottom, 255, -1)

    image_resize = cv2.resize(big_image,(width,height))
    #image_resize = cv2.GaussianBlur(image_resize,(5,5),0.6)
    mask_2 = image_resize.copy().astype(np.uint8)
    _, mask_2 = cv2.threshold(mask_2, min_intensity, 255, cv2.THRESH_BINARY)
    mask_2 += image_resize.astype(np.uint8)
    mask_2 = cv2.bitwise_not(mask_2)

    image_ret = cv2.bitwise_and(image, image, mask = mask_2)
    image_ret = cv2.add(image_ret, image_resize)
    return image_ret, mask, rect_top, rect_bottom

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

def sliding_window(image, image_mask, window_size, pad_h, pad_v):
    windows_h = (width - window_size) // pad_h + 1
    windows_v = (height - window_size) // pad_v + 1
    crops = np.zeros((windows_v, windows_h, window_size, window_size))
    labels = np.zeros((windows_v, windows_h))
    for j in range(windows_v):
        y_top = j*pad_v
        y_bottom = j*pad_v + window_size
        for i in range(windows_h):
            x_top = i*pad_h
            x_bottom = i*pad_h + window_size
            crop = image[y_top:y_bottom, x_top:x_bottom]
            crops[j,i,:,:] = crop
            mask_crop = image_mask[y_top:y_bottom, x_top:x_bottom]
            labels[j,i] = mask_crop.any()
    return crops, labels

def plot_masked(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    fig = plt.imshow(masked, vmin=0, vmax=255)
    pass

def plot_label(image, rect_top, rect_bottom):
    temp = image.copy()
    temp = cv2.rectangle(temp, rect_top, rect_bottom, 255, 1)
    fig = plt.imshow(temp, vmin=0, vmax=255)
    pass

def resize_labels(labels, pad_h, pad_v, window_size):
  labels_resize = np.zeros_like(image, dtype='uint8')

  obj = np.array(np.where(labels>0))
  pairs = obj.transpose()[:,[1, 0]]

  for pair in pairs:
    start = tuple(pair*[pad_h,pad_v] + [ window_size,0])
    end = tuple(pair*[pad_h,pad_v] + [0, window_size])
    cv2.rectangle(labels_resize, start, end, 1, -1)
  return labels_resize

def compare_labels(true_labels, sampled_labels):
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
  ax.set_xticks(np.arange(0, width, pad_h), minor=True)
  ax.set_yticks(np.arange(0, height, pad_v), minor=True)
  ax.set_xticks(np.arange(0, width, pad_h*4))
  ax.set_yticks(np.arange(0, height, pad_v*4))

  plt.imshow(true_labels, vmin=0, vmax=255)
  plt.imshow(sampled_labels, vmin=0, vmax=1, alpha=0.5)
  # And a corresponding grid
  ax.grid(which='both')
  #ax.grid(which='minor', alpha=0.5, color='black')
  #ax.grid(which='major', alpha=0.5, color='black')
  pass
