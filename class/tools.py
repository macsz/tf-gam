def get_color_intensity(prob, norm=False, threshold_down=0.5, threshold_up=0.8):
    max_val = 0.4
    min_val = 0.1
    if not norm:
        val = max_val * prob
    else:
        val = ((max_val - min_val) * (prob - threshold_down)) / (threshold_up - threshold_down) + min_val
    return val


def get_cached_prob(cache, frame, h, r):
    """

    :param cache: list of image caches
    :param frame: frame id
    :param h:
    :param r:
    :return:
    """
    if not cache:
        return
    active_frames = 7
    threshold = 3
    activity = 0
    start_frame = frame - active_frames
    if start_frame < 0:
        start_frame = 0
    for i in range(start_frame, frame):
        if cache[i][h][r]:
            activity += 1
    return activity >= threshold
