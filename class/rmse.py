import sys
import math

if len(sys.argv) == 1:
    print('Error. Missing param: provide path to log file.')
    exit()

path = sys.argv[1]

static_avg = 0.0
sqrs_sum_active = 0.0
sqrs_sum_static = 0.0
frames = 0


with open(path) as f:
    for line in f.readlines():
        line = line.rstrip()
        if 'Static frame\'s average color for active cells' in line:
            static_avg = float(line.split(' ')[-1])
        elif 'Face (frame) average color for active cells' in line:
            frames += 1

            frame_avg_active = float(line.split(' ')[-1])
            sqr_active = math.pow(static_avg - frame_avg_active, 2)
            sqrs_sum_active += sqr_active

        elif 'Face (frame) static cells avg color' in line:
            frame_avg_static = float(line.split(' ')[-1])
            sqr_static = math.pow(static_avg - frame_avg_static, 2)
            sqrs_sum_static += sqr_static

rmse_active = math.sqrt(sqrs_sum_active/frames)
rmse_static = math.sqrt(sqrs_sum_static/frames)
print('frames\t\t{0}'.format(frames))
print('rmse_active\t{0}'.format(rmse_active))
print('rmse_static\t{0}'.format(rmse_static))
print('diff\t\t{0}'.format(rmse_active-rmse_static))
