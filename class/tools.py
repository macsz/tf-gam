import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def get_color_intensity(prob, norm=False, threshold_down=0.5, threshold_up=0.8):
    max_val = 0.4
    min_val = 0.1
    if not norm:
        val = max_val * prob
    else:
        val = ((max_val - min_val) * (prob - threshold_down)) / (threshold_up - threshold_down) + min_val
    return val


def get_cached_prob(cache, h, r, active_frames=7, threshold=3):
    """

    :param threshold:
    :param active_frames:
    :param cache: list of image caches
    :param h:
    :param r:
    :return:
    """
    if not cache:
        return True, 0
    activity = 0
    last_frame = len(cache)-1
    start_frame = last_frame - active_frames
    if start_frame < 0:
        return True, 0
        # start_frame = 0
    for i in range(start_frame, last_frame):
        if cache[i][h][r]:
            activity += 1
    cached_prob = activity >= threshold
    return cached_prob, activity


def get_column_power(mat, c):
    power = 0
    for row in range(0, 8):
        power += 1 if mat[c][row] else 0
    return power


def get_avg_color(img):
    return np.average(img)


def array_slice(arr, x, y, w, h):
    return arr[y:y+h, x:x+w]


def draw_grid(img, save_path='playground/saved.jpg',
              open_path='playground/image.jpg',
              face_coords_static=None):
    plt.close('all')
    if not img:
        img = np.array(Image.open(open_path), dtype=np.uint8)
    fig, ax = plt.subplots()
    xp = float(80 / 8)
    yp = float(60 / 8)
    cells_avg_color = []
    for x in range(8):
        for y in range(8):
            tile = patches.Rectangle((xp * x, yp * y), xp, yp, linewidth=1,
                                     edgecolor='g', facecolor='none')
            text_x = xp * x + xp / 3
            text_y = yp * y + yp / 2
            ax.annotate(
                '{0}:{1}'.format(x, y),
                xytext=(text_x, text_y),
                xy=(text_x, text_y), color='red'
            )
            ax.add_patch(tile)

            if face_coords_static:
                cells_avg_color.append(
                    get_avg_color(
                        array_slice(img,
                                    x=int(xp * x), w=int(80 / 8),
                                    y=int(yp * y), h=int(60 / 8)
                                    )
                    )
                )
    frame_active_avg = np.average(cells_avg_color)
    print('Static frame\'s average color for active cells:',
          frame_active_avg)
    ax.imshow(img)
    plt.axis("off")
    # plt.show()
    plt.savefig(save_path)


def convert_coords(coords_str):
    """
    Converts string coordinates into the tuple of ints
    :param coords_str: X1:Y1xX2:Y2
    :return: {x1, y1, x2, y2}
    """
    p1 = coords_str.split('x')[0]
    x1 = p1.split(':')[0]
    y1 = p1.split(':')[1]
    p2 = coords_str.split('x')[1]
    x2 = p2.split(':')[0]
    y2 = p2.split(':')[1]
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
