import sys
import math

if len(sys.argv) == 1:
    print('Error. Missing param: provide path to log file.')
    exit()

path = sys.argv[1]

static_avg = 0.0
sqrs_sum = 0.0
frames = 0


with open(path) as f:
    for line in f.readlines():
        line = line.rstrip()
        if 'Static frame\'s average color for active cells' in line:
            static_avg = float(line.split(' ')[-1])
            print('static_avg=', static_avg)
        elif 'Face (frame) average color for active cells' in line:
            frames += 1
            frame_avg = float(line.split(' ')[-1])
            sqr = math.pow(static_avg - frame_avg, 2)
            sqrs_sum += sqr

rmse = math.sqrt(sqrs_sum/frames)
print('frames=', frames)
print('rmse=', rmse)
