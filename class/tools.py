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
