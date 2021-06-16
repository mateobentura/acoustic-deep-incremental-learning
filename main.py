from ImageSynthetique import *


height = 300
width = 640


starting_point = np.array([300, 30])
spacing = 15
length = 12
lines = 15

image, mask = create_background(height, width, background=2)
image, mask = add_bars(image, mask, starting_point, spacing, length, lines)

#plot_masked(image, mask)
plt.imshow(image)
plt.savefig('test')
