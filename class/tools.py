def get_color_intensity(prob, norm=False, threshold_down=0.5, threshold_up=0.8):
    max_val = 0.4
    min_val = 0.1
    if not norm:
        val = max_val * prob
    else:
        val = ((max_val - min_val) * (prob - THRESHOLD_DOWN)) / (THRESHOLD_UP - THRESHOLD_DOWN) + min_val
    return val
