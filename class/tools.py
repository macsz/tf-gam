def get_color_intensity(prob, norm=False, threshold_down=0.5, threshold_up=0.8):
    max_val = 0.4
    min_val = 0.1
    if not norm:
        val = max_val * prob
    else:
        val = ((max_val - min_val) * (prob - threshold_down)) / (threshold_up - threshold_down) + min_val
    return val


def get_cached_prob(cache, h, r):
    """

    :param cache: list of image caches
    :param h:
    :param r:
    :return:
    """
    if not cache:
        return True
    active_frames = 7
    threshold = 3
    activity = 0
    last_frame = len(cache)
    start_frame = last_frame - active_frames
    if start_frame < 0:
        return True
        # start_frame = 0
    for i in range(start_frame, last_frame):
        try:
            if cache[i][h][r]:
                activity += 1
        except IndexError as ie:
            raise IndexError('Failed to get index:', i, h, r)

    return activity >= threshold


def get_column_power(mat, c):
    power = 0
    for row in range(0, 8):
        power += mat[c][row]
    return power
