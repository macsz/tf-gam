import sys
import math

if len(sys.argv) == 1:
    print('Error. Missing param: provide path to log file.')
    exit()

path = sys.argv[1]

static_avg = 0.0
sqrs_sum_active = 0.0
sqrs_sum_static = 0.0
frames_active = 0
frames_static = 0


with open(path) as f:
    for line in f.readlines():
        line = line.rstrip()
        if 'Static frame\'s average color for active cells' in line:
            static_avg = float(line.split(' ')[-1])
            static_avg /= 255
        elif 'Face (frame) average color for active cells' in line:
            frame_avg_active = float(line.split(' ')[-1])
            if math.isnan(frame_avg_active):
                continue

            frame_avg_active /= 255
            sqr_active = math.pow(static_avg - frame_avg_active, 2)
            sqrs_sum_active += sqr_active
            frames_active += 1

        elif 'Face (frame) static cells avg color' in line:
            frame_avg_static = float(line.split(' ')[-1])
            if math.isnan(frame_avg_static):
                continue

            frame_avg_static /= 255
            sqr_static = math.pow(static_avg - frame_avg_static, 2)
            sqrs_sum_static += sqr_static
            frames_static += 1

rmse_active = math.sqrt(sqrs_sum_active/frames_active)
rmse_static = math.sqrt(sqrs_sum_static/frames_static)
print('frames_active\t{0}'.format(frames_active))
print('frames_static\t{0}'.format(frames_static))
print('rmse_active\t{0}'.format(rmse_active))
print('rmse_static\t{0}'.format(rmse_static))
print('diff\t\t{0}'.format(rmse_active-rmse_static))
