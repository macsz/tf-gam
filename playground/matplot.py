import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

plt.close('all')

image_path = 'playground/image.jpg'

img_full = np.array(Image.open(image_path), dtype=np.uint8)

fig, ax = plt.subplots()

xp = float(80/8)
yp = float(60/8)

for x in range(8):
    for y in range(8):
        tile = patches.Rectangle((xp * x, yp * y), xp, yp, linewidth=1,
                                 edgecolor='g', facecolor='none')
        text_x = xp * x + xp/3
        text_y = yp * y + yp/2
        ax.annotate(
            '{0}:{1}'.format(x, y),
            xytext=(text_x, text_y),
            xy=(text_x, text_y), color='red'
        )
        ax.add_patch(tile)

ax.imshow(img_full)
plt.axis("off")
plt.show()

# img_full = np.array(Image.open(image_path), dtype=np.uint8)
# print(len(img_full), len(img_full[0]))
# fig, ax = plt.subplots(1)
# lum_img = img_full[:,:,0]
# ax.imshow(lum_img, cmap="hot")
#
# plt.axis("off")
# plt.show()
# Generate some data :

# N = 5
# x, y = np.random.rand(N), np.random.rand(N)
# w, h = np.random.rand(N)/10 + 0.05, np.random.rand(N)/10 + 0.05
# colors = np.vstack([np.random.random_integers(0, 255, N),
#                     np.random.random_integers(0, 255, N),
#                     np.random.random_integers(0, 255, N)]).T
#
# # Plot and draw the data :
#
# fig = plt.figure(figsize=(7, 7), facecolor='white')
# ax = fig.add_subplot(111, aspect='equal')
# for i in range(N):
#     ax.add_patch(patches.Rectangle((x[i], y[i]), w[i], h[i], fc=colors[i]/255., ec='none'))
# ax.axis([0, 1, 0, 1])
# ax.axis('off')
# fig.canvas.draw()
#
# # Save data in a rgb string and convert to numpy array :
#
# rgb_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# rgb_data = rgb_data.reshape((int(len(rgb_data)/3), 3))
#
# # Keep only unique colors :
#
# print(len(rgb_data))
#
# rgb_data = np.vstack({tuple(row) for row in rgb_data})
#
# # Show and save figure :
#
# fig.savefig('rectangle_colors.png')
# plt.show()
